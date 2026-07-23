/* Recis Module: recis monitor Sock Writer
 */
#include "column-io/monitor/sock_writer.h"

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

int RECIS_MONITOR_SOCK_WRITER::write(const char* body) {
  int sockfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (sockfd < 0) {
    fprintf(stderr, "[WARN] [%s:%d] socket create fail\n", __FILE__, __LINE__);
    return -1;
  }
  timeval tv{};
  tv.tv_sec = timeout_ms_;
  tv.tv_usec = 0;
  if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
    fprintf(stderr, "[WARN] [%s:%d] setsockopt RCVTIMEO fail\n", __FILE__,
            __LINE__);
  }
  if (setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0) {
    fprintf(stderr, "[WARN] [%s:%d] setsockopt SNDTIMEO fail\n", __FILE__,
            __LINE__);
  }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, sock_path_.c_str(), sizeof(addr.sun_path) - 1);
  if (::connect(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    // fprintf(stderr, "[TRACE] [%s:%d] socket conn fail:%s\n", __FILE__,
    //     __LINE__, sock_path_.c_str());
    ::close(sockfd);
    return -1;
  }
  std::string request = "POST / HTTP/1.1\r\nHost: local\r\n";
  request += "Content-Type: text/plain\r\nContent-Length: ";
  request += std::to_string(strlen(body)) + "\r\n\r\n";
  request += body;

  ssize_t sent = ::write(sockfd, request.data(), request.size());
  char resp[256];
  recv(sockfd, resp, sizeof(resp), 0);  // prevent server EPIPE
  if (sent < 0) {
    fprintf(stderr, "[INFO] [%s:%d] socket write fail:%s\n", __FILE__, __LINE__,
            sock_path_.c_str());
    ::close(sockfd);
    return -1;
  }
  if (static_cast<size_t>(sent) < request.size()) {
    // fprintf(stderr, "[TRACE] [%s:%d] drop rest bytes since %ld of %ld\n",
    //     __FILE__, __LINE__, sent, request.size());
  }
  ::close(sockfd);
  return 0;
}
