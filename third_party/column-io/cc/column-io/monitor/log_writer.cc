#include "column-io/monitor/log_writer.h"

#include <ctype.h>
#include <libgen.h>
#include <pthread.h>
#include <sys/stat.h>

#define __recis_monitor_flock(fp)                                            \
  do {                                                                       \
    int fd = fileno(fp);                                                     \
    if (fd < 0 || flock(fd, LOCK_EX) < 0) {                                  \
      fprintf(stderr, "[ERROR] [%s:%d] flock fail:%d\n", __FILE__, __LINE__, \
              fd);                                                           \
      return -1;                                                             \
    }                                                                        \
  } while (0)

#define __recis_monitor_funlock(fp)                                            \
  do {                                                                         \
    int fd = fileno(fp);                                                       \
    if (fd < 0 || flock(fd, LOCK_UN) < 0) {                                    \
      fprintf(stderr, "[ERROR] [%s:%d] funlock fail:%d\n", __FILE__, __LINE__, \
              fd);                                                             \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static time_t __recis_monitor_get_slottime(time_t t,
                                           RECIS_MONITOR_LOG_ROTATE_T interval);
static void __recis_monitor_fmt_pathname(const char *prefix, time_t slot,
                                         char *out, size_t out_size);
static int __recis_monitor_unlink_olds(const char *soft_path);
static char *__recis_monitor_relative_path_calc(const char *relative_dir,
                                                const char *new_fullpath,
                                                size_t size);
static int __recis_monitor_safe_rotate_file(RECIS_MONITOR_LOG_WRITER_T *lw);

static int __recis_monitor_log_atfork_lw_idx = 0;
static RECIS_MONITOR_LOG_WRITER_T *__recis_monitor_atfork_lws[32];
static void __recis_monitor_log_atfork_prep() {
  for (int i = 0; i < __recis_monitor_log_atfork_lw_idx; i++)
    pthread_mutex_lock(&(__recis_monitor_atfork_lws[i]->mutex));
}
static void __recis_monitor_log_atfork_parent() {
  for (int i = 0; i < __recis_monitor_log_atfork_lw_idx; i++)
    pthread_mutex_unlock(&(__recis_monitor_atfork_lws[i]->mutex));
}
static void __recis_monitor_log_atfork_child() {
  for (int i = 0; i < __recis_monitor_log_atfork_lw_idx; i++) {
    pthread_mutex_destroy(&(__recis_monitor_atfork_lws[i]->mutex));
    pthread_mutex_init(&(__recis_monitor_atfork_lws[i]->mutex), NULL);
  }
}
void recis_monitor_log_atfork_setup(RECIS_MONITOR_LOG_WRITER_T *lw) {
  if (!lw || __recis_monitor_log_atfork_lw_idx >= 32) {
    return;
  }
  __recis_monitor_atfork_lws[__recis_monitor_log_atfork_lw_idx] = lw;
  if (__recis_monitor_log_atfork_lw_idx == 0) {
    pthread_atfork(__recis_monitor_log_atfork_prep,
                   __recis_monitor_log_atfork_parent,
                   __recis_monitor_log_atfork_child);
  }
  __recis_monitor_log_atfork_lw_idx++;
}

static time_t __recis_monitor_get_slottime(
    time_t t, RECIS_MONITOR_LOG_ROTATE_T interval) {
  return (t / interval) * interval;
}
static void __recis_monitor_fmt_pathname(const char *prefix, time_t slot,
                                         char *dst, size_t dst_size) {
  struct tm tm = {0};
  if (localtime_r(&slot, &tm) == NULL) {
    return;
  }
  snprintf(dst, dst_size, "%s.%04d%02d%02d_%02d%02d%02d", prefix,
           tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min,
           tm.tm_sec);
}
static int __recis_monitor_mkdir_p(const char *soft_path) {
  char path[512] = {0};
  strncpy(path, soft_path, sizeof(path) - 1);
  size_t begin = (path[0] == '/' ? 1 : 0);
  size_t len = strlen(path);
  while (begin < len) {
    char *slash = strchr(path + begin, '/');
    if (!slash) {
      break;
    }
    *slash = '\0';
    // printf("[DEBUG] [%s:%d] mkdir %s\n", __FILE__, __LINE__, path);
    int ret = mkdir(path, 0755);
    if (ret != 0 && errno != EEXIST) {
      fprintf(stderr, "[ERROR] [%s:%d] mkdir fail for %s\n", __FILE__, __LINE__,
              path);
      return -1;
    }
    *slash = '/';
    begin = (slash - path) + 1;
  }

  struct stat st;
  if (access(path, F_OK) != 0) {
    return -1;
  }
  if (stat(path, &st) != 0) {
    return -1;
  }
  return S_ISDIR(st.st_mode);
  ;
}
static int __recis_monitor_unlink_olds(const char *soft_path) {
  char path[256];
  strncpy(path, soft_path, sizeof(path));
  char *slash = strrchr(path, '/');
  if (slash) {
    *slash = '\0';
  } else {
    strcpy(path, ".");
  }

  DIR *dir = opendir(path);
  if (!dir) return -1;

  // get prefix: the base_dir/ or root index
  struct dirent *ent;
  const char *prefix = strrchr(soft_path, '/');
  prefix = prefix ? (prefix + 1) : soft_path;

  // path(256) + '/'(1) + path(256)
  enum { __PATH_MAX = 256 + 1 + 256 };
  struct {
    char name[__PATH_MAX];
    time_t mtime;
  } newest[RECIS_MONITOR_LOG_KEEP_NUM];
  int kept = 0;

  while ((ent = readdir(dir)) != NULL) {
    // filter file than the filename == prefix + '/' + "d_name.log" + ".2025*"
    // ... << dot << 4 << \0 << 256
    if (strncmp(ent->d_name, prefix, strlen(prefix)) != 0) continue;
    const char *dot = strrchr(ent->d_name, '.');
    if (!dot || (dot - ent->d_name) > 255 - 4 - 1) continue;

    const char *suffix = dot + 1;
    if (!(suffix[0] == '2' && suffix[1] == '0' &&
          isdigit((unsigned char)suffix[2]) &&
          isdigit((unsigned char)suffix[3]))) {
      continue;
    }

    char full[__PATH_MAX];
    int r = snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
    if (r < 0 || r >= (int)sizeof(full)) continue;

    struct stat st;
    if (stat(full, &st) != 0) continue;

    if (kept < RECIS_MONITOR_LOG_KEEP_NUM) {
      snprintf(newest[kept].name, sizeof(newest[kept].name), "%s", full);
      newest[kept].mtime = st.st_mtime;
      kept++;
    } else {  // find the oldest file reserving in disk now
      int min_i = 0;
      time_t min_t = newest[0].mtime;
      for (int i = 1; i < kept; ++i) {
        if (newest[i].mtime < min_t) {
          min_t = newest[i].mtime;
          min_i = i;
        }
      }
      if (st.st_mtime <= min_t) {
        unlink(full);  // me is the oldest, del me
        continue;
      }
      unlink(newest[min_i].name);
      snprintf(newest[min_i].name, sizeof(newest[min_i].name), "%s", full);
      newest[min_i].mtime = st.st_mtime;
    }
  }
  closedir(dir);
  return 0;
}
static char *__recis_monitor_relative_path_calc(const char *base_dir,
                                                const char *target_path,
                                                size_t size) {
  char abs_base[size];
  char abs_target[size];
  if (!realpath(base_dir, abs_base)) {
    return NULL;
  }
  if (!realpath(target_path, abs_target)) {
    return NULL;
  }

  // Split both paths into components
  char *base_parts[size / 2];
  char *target_parts[size / 2];
  int nb = 0, nt = 0, common = 0;

  char *tmp_base = strdup(abs_base);
  char *tmp_target = strdup(abs_target);
  for (char *p = strtok(tmp_base, "/"); p; p = strtok(NULL, "/")) {
    base_parts[nb++] = p;
  }
  for (char *p = strtok(tmp_target, "/"); p; p = strtok(NULL, "/")) {
    target_parts[nt++] = p;
  }
  // Find common prefix
  while (common < nb && common < nt &&
         strcmp(base_parts[common], target_parts[common]) == 0) {
    common++;
  }
  // Build relative path
  char rel[size];
  memset(rel, 0, size);
  for (int i = common; i < nb; i++) {
    strcat(rel, "../");
  }
  for (int i = common; i < nt; i++) {
    strcat(rel, target_parts[i]);
    if (i + 1 < nt) {
      strcat(rel, "/");
    }
  }
  free(tmp_base);
  free(tmp_target);
  return strdup(rel);
}

static int __recis_monitor_safe_rotate_file(RECIS_MONITOR_LOG_WRITER_T *lw) {
  FILE *old_fp = NULL;
  if (lw->fp) {
    old_fp = lw->fp;
    __recis_monitor_flock(old_fp);
  }

  char new_fullpath[512];
  __recis_monitor_fmt_pathname(lw->soft_path, lw->cur_slot, new_fullpath,
                               sizeof(new_fullpath));
  // printf("[DEBUG] [%s:%d] rotate create new_fullpath:%s\n", __FILE__,
  // __LINE__, new_fullpath);
  lw->fp = fopen(new_fullpath, "a");
  if (!lw->fp) {
    fprintf(stderr, "[ERROR] [%s:%d] fopen rotate create fail: %s\n", __FILE__,
            __LINE__, new_fullpath);
    if (old_fp) __recis_monitor_funlock(old_fp);
    return -1;
  }
  __recis_monitor_flock(lw->fp);

  // touch/update symbol link xxx.log -> newest timestamp filepath
  char relative_dir[512];
  strncpy(relative_dir, lw->soft_path, sizeof(relative_dir));
  strcpy(relative_dir, dirname(relative_dir));

  char *link_rel =
      __recis_monitor_relative_path_calc(relative_dir, new_fullpath, 512);
  if (!link_rel) {
    fprintf(stderr, "[ERROR] [%s:%d] calc relative_path fail: %s\n", __FILE__,
            __LINE__, new_fullpath);
    if (old_fp) __recis_monitor_funlock(old_fp);
    __recis_monitor_funlock(lw->fp);
    return -1;
  }

  unlink(lw->soft_path);                       // del old symbol link
  int ret = symlink(link_rel, lw->soft_path);  // relink new symbol link
  free(link_rel);
  if (ret != 0) {
    fprintf(stderr, "[ERROR] [%s:%d] symlink new link fail: %s\n", __FILE__,
            __LINE__, new_fullpath);
    if (old_fp) __recis_monitor_funlock(old_fp);
    __recis_monitor_funlock(lw->fp);
    return -1;
  }

  // del old-fashioned rotated files
  __recis_monitor_unlink_olds(lw->soft_path);
  if (old_fp) {
    __recis_monitor_funlock(old_fp);
    fclose(old_fp);  // auto flush in fclose
  }
  __recis_monitor_funlock(lw->fp);
  return 0;
}

int recis_monitor_safe_flush(RECIS_MONITOR_LOG_WRITER_T *lw) {
  if (lw->buf_len == 0) return 0;
  __recis_monitor_flock(lw->fp);
  size_t written = fwrite(lw->buf, 1, lw->buf_len, lw->fp);
  if (written <= 0) {
    fprintf(stderr, "[WARNING] [%s:%d] flush failed on %ld, no bytes written\n",
            __FILE__, __LINE__, written);
  }
  fflush(lw->fp);
  lw->buf_len = 0;
  lw->last_write_time = time(NULL);
  __recis_monitor_funlock(lw->fp);
  return (written > 0) ? 0 : -1;
}

RECIS_MONITOR_LOG_WRITER_T *recis_monitor_create_logger(const char *base_path) {
  RECIS_MONITOR_LOG_WRITER_T *lw =
      (RECIS_MONITOR_LOG_WRITER_T *)malloc(sizeof(RECIS_MONITOR_LOG_WRITER_T));
  RECIS_MONITOR_LOG_ROTATE_T interval = RECIS_MONITOR_LOG_ROTATE_DEFAULT;

  int ret = recis_monitor_init_logger(lw, base_path, interval);
  if (ret != 0) {
    free(lw);
    return NULL;
  }
  return lw;
}

int recis_monitor_init_logger(RECIS_MONITOR_LOG_WRITER_T *lw,
                              const char *base_path,
                              RECIS_MONITOR_LOG_ROTATE_T interval) {
  if (lw == NULL || base_path == NULL) {
    return -1;
  }
  if (strlen(base_path) >= sizeof(lw->soft_path)) {
    return -1;
  }
  memset(lw, 0, sizeof(*lw));
  strncpy(lw->soft_path, base_path, sizeof(lw->soft_path) - 1);
  __recis_monitor_mkdir_p(lw->soft_path);
  time_t now = time(NULL);
  if (interval == RECIS_MONITOR_LOG_ROTATE_INVALID)
    interval = RECIS_MONITOR_LOG_ROTATE_DEFAULT;
  lw->cur_slot = __recis_monitor_get_slottime(now, interval);
  lw->interval = interval;
  lw->last_write_time = time(NULL);
  if (pthread_mutex_init(&lw->mutex, NULL) != 0) {
    return -1;
  }
  if (__recis_monitor_safe_rotate_file(lw) != 0) {
    pthread_mutex_destroy(&lw->mutex);
    return -1;
  }
  return 0;
}

int recis_monitor_logger_write(RECIS_MONITOR_LOG_WRITER_T *lw, const char *msg,
                               bool flush) {
  if (lw == NULL) {
    return -1;
  }
  if (msg == NULL) {
    return -1;
  }

  pthread_mutex_lock(&lw->mutex);
  time_t now = time(NULL);
  time_t slot = __recis_monitor_get_slottime(now, lw->interval);
  if (slot != lw->cur_slot) {
    recis_monitor_safe_flush(lw);
    lw->cur_slot = slot;
    if (__recis_monitor_safe_rotate_file(lw) != 0) {
      pthread_mutex_unlock(&lw->mutex);
      return -1;
    }
  } else if (now - lw->last_write_time > 1) {
    // NOTE: prevent too old msg, essential for metric log
    recis_monitor_safe_flush(lw);
  }

  size_t msglen = strlen(msg);
  if (msglen > RECIS_MONITOR_LOG_WRITER_BUF_MAX) {
    fprintf(stderr,
            "[WARNING] [%s:%d] drop msg chunck since msglen:%ld is too long\n",
            __FILE__, __LINE__, msglen);
    msglen = RECIS_MONITOR_LOG_WRITER_BUF_MAX;
  }
  if (lw->buf_len > 0 &&
      lw->buf_len + msglen >= RECIS_MONITOR_LOG_WRITER_BUF_MAX) {
    int ret = recis_monitor_safe_flush(lw);
    if (ret != 0) {
      pthread_mutex_unlock(&lw->mutex);
      return -1;
    }
  }
  memcpy(lw->buf + lw->buf_len, msg, msglen);
  lw->buf_len += msglen;
  if (msglen >= RECIS_MONITOR_LOG_WRITER_BUF_MAX) {
    int ret = recis_monitor_safe_flush(lw);
    if (ret != 0) {
      pthread_mutex_unlock(&lw->mutex);
      return -1;
    }
  }
  if (flush) {
    int ret = recis_monitor_safe_flush(lw);
    if (ret != 0) {
      pthread_mutex_unlock(&lw->mutex);
      return -1;
    }
  }
  lw->last_write_time = now;
  pthread_mutex_unlock(&lw->mutex);
  return 0;
}

int recis_monitor_close_logger(RECIS_MONITOR_LOG_WRITER_T *lw) {
  if (lw == NULL) return -1;
  pthread_mutex_lock(&lw->mutex);

  if (lw->buf_len > 0 && recis_monitor_safe_flush(lw) != 0) {
    fprintf(stderr, "[WARNING] [%s:%d] failed to flush buffer on close\n",
            __FILE__, __LINE__);
  }
  lw->soft_path[0] = '\0';
  if (lw->fp) fclose(lw->fp);
  lw->fp = NULL;
  lw->cur_slot = 0;
  lw->buf_len = 0;

  pthread_mutex_unlock(&lw->mutex);
  pthread_mutex_destroy(&lw->mutex);
  return 0;
}