import argparse
import json
import os
import re
import subprocess

import requests
from packaging.version import Version


book_id = 161951469
# doc_id = 479478532  # 测试文档 doc_id = 515764655
pipeline_id = 36971
run_id = os.environ.get("AONE_CI_JOBRUN_UID", "").split("-", 1)[0]


def get_package_version():
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(
        os.path.abspath(os.path.join(pwd, "..", "..", "recis", "__init__.py"))
    ) as f:
        groups = re.findall(r"__version__.*([0-9]+)\.([0-9]+)\.([0-9]+)", f.read())
        main_version, minor_version, patch_version = groups[0]
        print(f"RecIS version {main_version}.{minor_version}.{patch_version}")
        version = f"{main_version}.{minor_version}.{patch_version}"
        return version


def parse_column_labels(column_labels_str):
    print("the column_labels_str raw message:\n{}\n".format(column_labels_str))
    column_labels_list = column_labels_str.strip().split("|")
    column_labels = [label.strip() for label in column_labels_list if label.strip()]
    return column_labels


def parse_package_message(package_message, column_labels):
    print("RAW package_message:\n{}\n".format(package_message))

    package_list = []

    if package_message.strip().startswith("[") and package_message.strip().endswith(
        "]"
    ):
        try:
            items = json.loads(package_message)
            # items: ["| nv | 0 | 128 | ... |", "| ppu | 1 | ... |"]
            package_strings = items
        except json.JSONDecodeError:
            # 不是标准 JSON，则 fallback 到旧格式
            package_strings = package_message.strip().strip("[]").split(",")
    else:
        package_strings = package_message.strip().strip("[]").split(",")

    # Now parse each entry
    for entry in package_strings:
        entry = entry.strip().strip('"').strip("'")
        fields = [x.strip() for x in entry.split("|") if x.strip()]

        if len(fields) != len(column_labels):
            print("❌ Invalid package entry!")
            print("Expected {} fields: {}".format(len(column_labels), column_labels))
            print("Got: {} fields → {}".format(len(fields), entry))
            continue

        package_list.append(fields)

    return package_list


def wget_check_urls(column_labels, package_list, timeout=5):
    if "URL" not in column_labels:
        raise ValueError('column_labels not include "URL"')

    url_idx = column_labels.index("URL")
    bad_urls_cnt = 0

    for row in package_list:
        if url_idx >= len(row):
            print(f"❌ Invalid package entry! {row}")
            bad_urls_cnt += 1
            continue
        url = (row[url_idx] or "").strip()
        if not url:
            print(f"❌ Invalid url! {url}")
            bad_urls_cnt += 1
            continue

        # --spider: 不下载文件，只探测
        # -q: 安静模式；--timeout: 超时；--tries=3: 重试3次
        cmd = ["wget", "--spider", "-q", "--timeout", str(timeout), "--tries", "3", url]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if r.returncode != 0:
            print(f"❌ [BAD URL] {url}")
            bad_urls_cnt += 1

    if bad_urls_cnt > 0:
        raise RuntimeError(f"URL check failed, bad count={bad_urls_cnt}")
    else:
        print("✅ All URLs are valid")


def get_yuque_docs(yuque_token, doc_id):
    url = "https://yuque-api.antfin-inc.com/api/v2/repos/docs/{}".format(doc_id)
    headers = {
        "X-Auth-Token": yuque_token,
    }
    response = requests.get(url, headers=headers)
    return response.json()


def get_recis_latest_version_lake(body_lake):
    # 匹配 lake 格式中的： RecIS X.X.X
    pattern = r"RecIS\s+([0-9]+\.[0-9]+\.[0-9]+)"
    versions = re.findall(pattern, body_lake)

    if not versions:
        raise ValueError("No RecIS version found in lake body")

    latest = max(versions, key=Version)
    return latest


def get_yuque_msg(yuque_docs):
    slug = yuque_docs["data"]["slug"]
    title = yuque_docs["data"]["title"]
    public = yuque_docs["data"]["public"]
    doc_format = yuque_docs["data"]["format"]
    body_lake = yuque_docs["data"]["body_lake"]
    recis_latest_version = get_recis_latest_version_lake(body_lake)
    return slug, title, public, doc_format, body_lake, recis_latest_version


def build_lake_table(package_list):
    # package_list: [[header...], [row1...], ...]

    header = package_list[0]
    rows = package_list[1:]

    html = []
    html.append("<h3><span>RecIS</span></h3>")
    url = f"https://code.alibaba-inc.com/xrec/xrec/ci/jobs?pipelineId={pipeline_id}&pipelineRunId={run_id}&createType=yaml"
    html.append(f'<p><a href="{url}" target="_blank"><span>{url}</span></a></p>')
    html.append('<table class="lake-table" margin="true">')
    html.append("<tbody>")

    # header row: background rgb(241,248,255)
    html.append("<tr>")
    for cell in header:
        html.append(
            f'<td style="background-color: rgb(241, 248, 255)"><p>'
            f'<strong><span class="lake-fontsize-11" style="color: rgb(51, 51, 51)">{cell}</span></strong>'
            f"</p></td>"
        )
    html.append("</tr>")

    # data rows
    for idx, row in enumerate(rows, start=1):
        if idx % 2 == 1:
            bg = "rgb(241, 248, 255)"  # 奇数行背景色
        else:
            bg = "rgb(246, 248, 250)"  # 偶数行背景色（语雀 lake 内置的白-灰交替）

        html.append("<tr>")
        for cell in row:
            html.append(
                f'<td style="background-color: {bg}"><p>'
                f'<span class="lake-fontsize-11" style="color: rgb(51, 51, 51)">{cell}</span>'
                f"</p></td>"
            )
        html.append("</tr>")

    html.append("</tbody></table>")
    return "".join(html)


def build_new_recis_h2_block(
    body_lake, package_list, recis_latest_version, column_labels
):
    # --- 生成新版本号 + 新表格 ---
    new_recis_version = get_package_version()
    package_list = [column_labels] + package_list
    new_table_html = build_lake_table(package_list)

    # --- 找到当前 RecIS h2 区块范围 ---
    h2_pattern = rf"<h2[^>]*?>\s*<span[^>]*?>\s*RecIS\s+{re.escape(recis_latest_version)}\s*</span>\s*</h2>"
    h2_match = re.search(h2_pattern, body_lake)
    if not h2_match:
        raise ValueError(
            f"Cannot find RecIS section for version: {recis_latest_version}"
        )

    h2_end = h2_match.end()

    next_h2_match = re.search(
        r"<h2[^>]*?>\s*<span[^>]*?>\s*RecIS\s+", body_lake[h2_end:]
    )
    h2_next_start = h2_end + next_h2_match.start() if next_h2_match else len(body_lake)

    region = body_lake[h2_end:h2_next_start]

    # --- 在该 h2 区块内取出完整 column-io 的 h3 区块 ---
    h3_pattern = r"<h3[^>]*?>\s*<span[^>]*?>\s*column-io\s*</span>\s*</h3>"
    h3_match = re.search(h3_pattern, region)
    if not h3_match:
        raise ValueError(
            "Cannot find column-io h3 block under the specified RecIS h2 block"
        )

    # h3 在 region 内的起点/终点（相对 region）
    h3_start = h3_match.start()
    column_io_h3_block = region[h3_start:h2_next_start]

    # --- 构造新的 h2 区块
    new_h2_html = f"<h2><span>RecIS {new_recis_version}</span></h2>"

    new_h2_block_html = new_h2_html + new_table_html + column_io_h3_block
    first_h2 = re.search(r"<h2\b", body_lake)
    insert_pos = first_h2.start() if first_h2 else len(body_lake)

    new_body_lake = body_lake[:insert_pos] + new_h2_block_html + body_lake[insert_pos:]

    return new_body_lake


def update_yuque_docs(
    yuque_token, book_id, doc_id, slug, title, public, doc_format, new_body
):
    print("raw format: {}".format(doc_format))
    url = "https://yuque-api.antfin-inc.com/api/v2/repos/{}/docs/{}".format(
        book_id, doc_id
    )
    # 设置请求头，包含 X-Auth-Token 以进行身份验证
    headers = {"X-Auth-Token": yuque_token, "Content-Type": "application/json"}

    data = {
        "slug": slug,
        "title": title,
        "public": public,
        "format": "lake",
        "body": new_body,
    }

    # response = requests.put(url, headers=headers, data=data)
    response = requests.put(url, headers=headers, json=data)
    if response.status_code == 200:
        print("Update yuque docs success!")
    else:
        print(response.text)
        raise ValueError("Update yuque docs failed, please check it!")
    return response.json()


def parse_args():
    parser = argparse.ArgumentParser(description="Make yuque docs.")
    parser.add_argument("--package_message", type=str, required=True)
    parser.add_argument("--column_labels_str", type=str, required=True)
    parser.add_argument("--doc_id", type=int, required=True)
    return parser.parse_args()


def main():
    print("\n" * 8)
    args = parse_args()
    yuque_token = os.environ.get("YUQUE_TOKEN")
    column_labels = parse_column_labels(args.column_labels_str)
    package_list = parse_package_message(args.package_message, column_labels)
    wget_check_urls(column_labels, package_list)
    yuque_docs = get_yuque_docs(yuque_token, args.doc_id)
    slug, title, public, doc_format, body_lake, recis_latest_version = get_yuque_msg(
        yuque_docs
    )
    print("latest recis version:\n{}\n".format(recis_latest_version))
    new_body = build_new_recis_h2_block(
        body_lake, package_list, recis_latest_version, column_labels
    )
    update_yuque_docs(
        yuque_token, book_id, args.doc_id, slug, title, public, doc_format, new_body
    )
    print("\n" * 8)


if __name__ == "__main__":
    main()
