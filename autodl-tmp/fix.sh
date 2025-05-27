#!/bin/bash

echo "===== OpenSSL 3.0 完整安装脚本 ====="
echo "此脚本将从源码编译并安装完整的OpenSSL 3.0.9"

# 创建一个新的目录用于编译
WORK_DIR=$(mktemp -d)
cd $WORK_DIR
echo "工作目录: $WORK_DIR"

# 下载OpenSSL源码
echo "正在下载OpenSSL 3.0.9源码..."
wget https://www.openssl.org/source/openssl-3.0.9.tar.gz
if [ $? -ne 0 ]; then
    echo "下载失败，尝试使用curl..."
    curl -O https://www.openssl.org/source/openssl-3.0.9.tar.gz
fi

# 解压源码
echo "正在解压源码..."
tar -xzf openssl-3.0.9.tar.gz
cd openssl-3.0.9

# 安装编译依赖
echo "正在安装编译依赖..."
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y build-essential checkinstall zlib1g-dev
elif command -v yum &> /dev/null; then
    yum groupinstall -y "Development Tools"
    yum install -y zlib-devel
fi

# 设置安装目录
INSTALL_DIR="/opt/openssl3"
echo "将安装OpenSSL 3.0.9到: $INSTALL_DIR"

# 配置编译选项
echo "配置OpenSSL编译选项..."
./config --prefix=$INSTALL_DIR --openssldir=$INSTALL_DIR/ssl shared zlib

# 编译
echo "开始编译OpenSSL (这可能需要几分钟)..."
make -j$(nproc)

# 安装
echo "安装OpenSSL..."
make install

# 验证安装
echo "验证OpenSSL安装..."
$INSTALL_DIR/bin/openssl version
if [ $? -ne 0 ]; then
    echo "安装验证失败!"
    exit 1
fi

# 配置环境变量
echo "配置环境变量..."

# 添加库路径到系统
echo "$INSTALL_DIR/lib" > /etc/ld.so.conf.d/openssl3.conf
ldconfig

# 创建所需的环境变量设置脚本
cat > /tmp/openssl3_env.sh << EOF
export PATH=$INSTALL_DIR/bin:\$PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH
export LIBRARY_PATH=$INSTALL_DIR/lib:\$LIBRARY_PATH
export C_INCLUDE_PATH=$INSTALL_DIR/include:\$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$INSTALL_DIR/include:\$CPLUS_INCLUDE_PATH
export OPENSSL_DIR=$INSTALL_DIR
EOF

# 应用环境变量
source /tmp/openssl3_env.sh

# 将环境变量添加到系统配置
cp /tmp/openssl3_env.sh /etc/profile.d/openssl3.sh

# 检查库是否可访问
echo "检查库文件是否可访问..."
if [ -f "$INSTALL_DIR/lib/libssl.so.3" ] && [ -f "$INSTALL_DIR/lib/libcrypto.so.3" ]; then
    echo "库文件存在!"
    ls -la $INSTALL_DIR/lib/libssl.so.3
    ls -la $INSTALL_DIR/lib/libcrypto.so.3
else
    echo "警告: 库文件未找到!"
    ls -la $INSTALL_DIR/lib/
fi

# 验证版本信息
echo "验证库版本信息..."
if [ -f "$INSTALL_DIR/lib/libcrypto.so.3" ]; then
    strings $INSTALL_DIR/lib/libcrypto.so.3 | grep -i "OPENSSL_3.0"
fi

# 重新安装torchdata
echo "重新安装Python包..."
pip uninstall -y torchdata
OPENSSL_DIR=$INSTALL_DIR pip install --force-reinstall torchdata

echo "设置当前会话的环境变量..."
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH

# 创建一个辅助脚本，供用户直接运行PyTorch程序
cat > torch_run.sh << EOF
#!/bin/bash
# 加载OpenSSL 3.0环境
source /etc/profile.d/openssl3.sh

# 将OpenSSL 3.0库路径添加到LD_PRELOAD
export LD_PRELOAD=$INSTALL_DIR/lib/libcrypto.so.3:$INSTALL_DIR/lib/libssl.so.3

# 运行命令行参数指定的程序
exec "\$@"
EOF

chmod +x torch_run.sh
cp torch_run.sh /usr/local/bin/

# 清理
echo "清理临时文件..."
cd /
rm -rf $WORK_DIR

echo "OpenSSL 3.0.9 安装完成!"
echo "要运行Python程序，建议使用:"
echo "  source /etc/profile.d/openssl3.sh"
echo "  export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
echo "或使用我们创建的辅助脚本:"
echo "  torch_run.sh python yourprogram.py" 