// MapDataSetSampleFilter — row-level denylist filter on sample_id.
//
// Mirrors MapDataSetK2RankCanXi (cc/column-io/dataset/map_dataset_k2_rank_canxi.cc)
// in shape: injects a `_sample_group_id` int64 column to the output schema
// (-1 = drop, 0 = keep) and lets the downstream Packer perform the actual
// row drop (packer.cc:459-470, do_classify path).
//
// Why we don't drop rows here: the column-mode batch is a list of CSR-like
// ragged tensors + indicator tensors; dropping a single row requires
// rewriting every ragged values/splits pair and every _indicator column.
// The Packer already implements exactly that under the `do_classify` branch.
// Reusing it cuts ~hundreds of lines and inherits its tested ragged/indicator
// invariants. (See AI_WIKI/dataset.md §4.1 and packer.cc::GetNextForGroup.)

#include "column-io/dataset/map_dataset_sample_filter.h"

#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"

namespace column {
namespace dataset {
namespace {

const std::string kDatasetName = "MapDataSetSampleFilter";

// Per-row sentinels matching packer's `_sample_group_id` contract.
// (packer.cc:459-461 skips any row whose group_id < 0; non-negative group_ids
// flow into per-group packers — we only use group 0.)
constexpr int64_t kSampleKeep = 0;
constexpr int64_t kSampleDrop = -1;

class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, const std::shared_ptr<DatasetBase> &input,
          const std::vector<
              std::map<std::string, std::vector<std::vector<int64_t>>>>
              &new_input_schema,
          const std::vector<
              std::map<std::string, std::vector<std::vector<int64_t>>>>
              &old_input_schema,
          const std::map<std::string, std::vector<std::string>> &filter_dict)
      : DatasetBase(name), input_(input), new_input_schema_(new_input_schema),
        old_input_schema_(old_input_schema) {
    // Build O(1)-lookup form of filter_dict ONCE at construction.
    // - absl::flat_hash_{map,set} (not std::unordered_*) for two reasons:
    //   1. Heterogeneous lookup: find/contains accept absl::string_view
    //      directly, so per-row parsing avoids the std::string allocation
    //      that std::unordered_map<string,...>::find(string_view) would force.
    //      With ~10 tokens per sample_id × batch_size × N batches in a
    //      streaming pipeline, this is a real allocator-pressure win.
    //   2. Better cache locality and lower per-key overhead than
    //      std::unordered_*.
    // ClassifySample is on the hot path (every row of every batch); these
    // constants are read concurrently across iterator threads without locks
    // (see filter_dict_ comment at class body).
    for (const auto &kv : filter_dict) {
      filter_dict_.emplace(kv.first, absl::flat_hash_set<std::string>(
                                          kv.second.begin(), kv.second.end()));
    }
  }

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::make_shared<Iterator>(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()),
        absl::StrCat(prefix, "::", name()));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    Iterator(const std::shared_ptr<Dataset> dataset, const std::string &prefix)
        : DatasetIterator<Dataset>({dataset, prefix}) {}

    Status Initialize() override {
      return this->dataset()->input_->MakeIterator(prefix(), &input_impl_);
    }

    Status
    GetNextInternal(std::vector<Tensor> *out_tensors, bool *end_of_sequence,
                    std::vector<size_t> *outputs_row_spliter = nullptr) override {
      std::lock_guard<std::mutex> l(mu_);
      if (!input_impl_) {
        // EOS protocol: OK + flag, NOT Status::OutOfRange.
        // See AI_WIKI.md §1 "三条核心约定" #1.
        *end_of_sequence = true;
        return Status::OK();
      }
      std::vector<Tensor> tmp_output;
      auto st = input_impl_->GetNext(&tmp_output, end_of_sequence);
      if (*end_of_sequence) {
        return Status::OK();
      }
      if (!st.ok()) {
        return st;
      }

      // Locate sample_id column. .at() throws on missing — the Python wrapper
      // (column_io/dataset/dataset.py::MapDatasetSampleFilter.__init__) validates
      // the schema up front, so reaching here without "sample_id" is a bug.
      int64_t sample_id_pos =
          this->dataset()->old_input_schema_[0].at("sample_id")[0][0];
      const Tensor sample_id = tmp_output[sample_id_pos];

      // Allocate _sample_group_id with the same row count as sample_id.
      // Assumption: sample_id is a 1-D string tensor whose length equals the
      // logical batch row count — this is the upstream contract enforced by
      // lake/odps readers (and matches map_dataset_k2_rank_canxi.cc:63).
      Tensor sample_group_id = Tensor(kInt64, sample_id.Shape());
      ClassifyBatch(sample_id, sample_group_id);

      // Narrowing int64_t (schema position) -> int32_t mirrors K2 baseline
      // (map_dataset_k2_rank_canxi.cc:67). Safe in practice — no realistic
      // schema exceeds 2B columns. Kept consistent for code-archaeology.
      int32_t group_id_t_pos = static_cast<int32_t>(
          this->dataset()->new_input_schema_[0].at("_sample_group_id")[0][0]);

      // Pass-through input tensors, then insert _sample_group_id at the
      // schema-decreed position. Same shape op as map_dataset_k2_rank_canxi.cc:69-75.
      out_tensors->clear();
      out_tensors->resize(tmp_output.size());
      for (size_t i = 0; i < tmp_output.size(); ++i) {
        (*out_tensors)[i] = tmp_output[i];
      }
      out_tensors->insert(out_tensors->begin() + group_id_t_pos,
                          sample_group_id);
      return Status::OK();
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      // Only input_impl_ is checkpointed. filter_dict_ is construction-time
      // immutable and intentionally NOT saved (it's reconstructed when the
      // upstream code re-builds this Dataset). Same as map_dataset_k2_rank_canxi.
      std::lock_guard<std::mutex> l(mu_);
      if (input_impl_) {
        RETURN_IF_ERROR(writer->WriteString(fullname("input"), ""));
        RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      if (reader->Contain(fullname("input"))) {
        RETURN_IF_ERROR(RestoreInput(reader, input_impl_));
      }
      return Status::OK();
    }

  private:
    // Per-batch fan-out over rows. Hot-path: O(row_count * tokens_per_row).
    void ClassifyBatch(const Tensor &sample_id, Tensor &sample_group_id) {
      int64_t item_num = sample_id.NumElements();
      std::string *sample_id_vec = sample_id.Raw<std::string>();
      int64_t *sample_group_id_vec = sample_group_id.Raw<int64_t>();
      // Empty filter dict fast-path: skip per-row parsing entirely.
      // memset is valid here because kSampleKeep == 0 and a zero-byte fill of
      // int64_t storage yields int64_t(0); avoids the loop's branch overhead
      // on large batches.
      if (this->dataset()->filter_dict_.empty()) {
        std::memset(sample_group_id_vec, 0,
                    static_cast<size_t>(item_num) * sizeof(int64_t));
        return;
      }
      for (int64_t i = 0; i < item_num; ++i) {
        sample_group_id_vec[i] = ClassifySample(sample_id_vec[i]);
      }
    }

    // Parse a single sample_id string and decide keep/drop.
    //
    // Format contract (producer-side):
    //   "<prefix>\x01<k1>:<v1>,<k2>:<v2>,...,<kN>:<vN>"
    // The leading prefix segment before \x01 is intentionally ignored
    // (matches map_dataset_k2_rank.cc:157-163's tokens[1] convention).
    //
    // Robustness vs the K2 baseline (map_dataset_k2_rank.cc:165):
    //   - Use absl::MaxSplits(':', 1) instead of plain StrSplit(':'). The
    //     baseline silently drops any value that itself contains ':' (e.g.
    //     an embedded timestamp like "ts:2026-01-01T12:34:56"). For a
    //     denylist that's a false-negative (we MISS a drop), which is the
    //     dangerous direction — MaxSplits keeps the whole value.
    //   - Any parse failure → return kSampleKeep ("default to keep") because
    //     denylist semantics favor false-negatives over false-positives
    //     when the data is malformed.
    //
    // OR-across-keys: any single key whose parsed value is in its denylist
    // triggers a drop; we short-circuit on the first hit.
    int64_t ClassifySample(const std::string &sample_id) {
      try {
        absl::string_view stripped = absl::StripAsciiWhitespace(sample_id);
        std::vector<absl::string_view> head_tail =
            absl::StrSplit(stripped, absl::MaxSplits('\x01', 1));
        if (head_tail.size() < 2) {
          // No \x01 separator → format not recognized → safe-side keep.
          return kSampleKeep;
        }
        const auto &filter_dict = this->dataset()->filter_dict_;
        for (absl::string_view token : absl::StrSplit(head_tail[1], ',')) {
          // Robust split-and-detect: a colon-less token must NOT be treated
          // as key=token,value="" — otherwise a user denylist that happens
          // to contain "" under a key that matches the standalone token
          // would erroneously drop the row. Use string_view::find(':') so
          // we can distinguish "no colon" from "leading colon".
          const auto colon_pos = token.find(':');
          if (colon_pos == absl::string_view::npos) {
            continue;  // genuinely malformed token (no ':' at all)
          }
          absl::string_view key = token.substr(0, colon_pos);
          absl::string_view value = token.substr(colon_pos + 1);
          if (key.empty()) {
            continue;  // leading-colon token (":value") — also malformed
          }
          // Heterogeneous lookup: absl::flat_hash_map/set accept string_view
          // directly, avoiding the per-row std::string allocation that the
          // std::unordered_* equivalents would force.
          auto it = filter_dict.find(key);
          if (it == filter_dict.end()) {
            continue;
          }
          if (it->second.contains(value)) {
            return kSampleDrop;
          }
        }
      } catch (const std::exception &) {
        // Defensive — none of the operations above are documented to throw,
        // but the K2 baseline (map_dataset_k2_rank.cc:174) also wraps in
        // try/catch and we want the same fail-safe.
      }
      return kSampleKeep;
    }

    std::mutex mu_;
    std::shared_ptr<IteratorBase> input_impl_;
  };

  std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>
      new_input_schema_;
  std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>
      old_input_schema_;
  std::shared_ptr<DatasetBase> input_;
  // O(1) lookup form built at construction. NEVER mutated post-ctor — readers
  // (Iterator::ClassifySample) need no lock. absl::flat_hash_* (not std::*)
  // because flat_hash supports heterogeneous lookup with absl::string_view,
  // letting the hot path skip per-row std::string construction.
  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>
      filter_dict_;
};

} // namespace

std::shared_ptr<dataset::DatasetBase> MapDataSetSampleFilter::MakeDataSet(
    const std::shared_ptr<DatasetBase> &input,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>
        &new_input_schema,
    const std::vector<std::map<std::string, std::vector<std::vector<int64_t>>>>
        &old_input_schema,
    const std::map<std::string, std::vector<std::string>> &filter_dict) {
  return std::make_shared<Dataset>(kDatasetName, input, new_input_schema,
                                   old_input_schema, filter_dict);
}

} // namespace dataset
} // namespace column
