# ColumnIo - A python lib for building pipeline to read columnar data.

## 构建镜像
```
docker build --network=host --no-cache --cpuset-cpus=0-95 -t ${image_name}
```

## 下载编译依赖
```
# 推荐在容器内安装依赖并编译columnIO
yum install -y t_search_kmonitor_client alog-devel autil-devel -b current #旧编译标准需要 
pip install cmake
```


## 编译及安装

`bash -x ./build_and_install.sh`

编译选项:
- `export NEED_ODPS_COLUMN=1` default:1  cxx11_abi1开关(关闭后ABI=0 直读algo模块将启用) (旧语义下为使用ODPS-storage, 现今语义下只用作CXX11ABI)
- `export NEED_CPU_ONLY=1` default:0  cpu-only开关(开启cpuonly模式后将构建只使用纯cpu能力的版本, 禁用cuda&torch等非cpu依赖. 适用于行读场景). 待实现
- `export NEED_ROW_ONLY=1` default:0  row-only开关(开启rowonly模式后将构建只使用纯行读能力的轻量化版本, 禁用列读sdk. 适用于纯行读场景). 待实现

## 测试
```
bash -x ./test.sh
```
