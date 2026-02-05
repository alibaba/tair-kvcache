# change from:
# https://github.com/deepseek-ai/3FS/blob/main/.github/workflows/build.yml
# https://github.com/deepseek-ai/3FS/blob/main/dockerfile/dev.ubuntu2004.dockerfile

# Only use rpm to pack binary files, not for installation
mkdir -p /tmp/hf3fs-dependencies/
mkdir -p /tmp/hf3fs-rpm-build/
mkdir -p /tmp/hf3fs-rpm-build/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
cp ./open_source/package/hf3fs-usrbio.spec /tmp/hf3fs-rpm-build/SPECS

pushd /tmp/hf3fs-dependencies/ || exit 1
  apt-get update
  DEBIAN_FRONTEND=noninteractive \
  apt-get install --no-install-recommends -y cmake libuv1-dev liblz4-dev liblzma-dev libdouble-conversion-dev \
  libdwarf-dev libunwind-dev git wget software-properties-common  libaio-dev libgflags-dev \
  libgoogle-glog-dev libgtest-dev libgmock-dev libgoogle-perftools-dev google-perftools libssl-dev \
  libboost-all-dev meson build-essential rpm

  wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key |tee /etc/apt/trusted.gpg.d/llvm.asc &&\
  add-apt-repository -y "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" &&\
  apt-get update && apt-get install -y clang-format-14 clang-14 clang-tidy-14 lld-14 libclang-rt-14-dev gcc-10 g++-10

  FDB_ARCH_SUFFIX=$(dpkg --print-architecture) && \
  case "${FDB_ARCH_SUFFIX}" in \
    amd64) export FDB_ARCH_SUFFIX=amd64 ;; \
    arm64) export FDB_ARCH_SUFFIX="aarch64" ;; \
    *) echo "Unsupported architecture: ${FDB_ARCH_SUFFIX}"; exit 1 ;; \
  esac
  wget https://github.com/apple/foundationdb/releases/download/7.1.61/foundationdb-clients_7.1.61-1_${FDB_ARCH_SUFFIX}.deb && dpkg -i foundationdb-clients_7.1.61-1_${FDB_ARCH_SUFFIX}.deb

  git clone https://github.com/libfuse/libfuse.git libfuse -b fuse-3.16.2 --depth 1 && mkdir libfuse/build && cd libfuse/build && CC=gcc-10 CXX=g++-10 meson setup .. && ninja &&  ninja install && cd ../.. && rm -rf libfuse

  mkdir -p {BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}

  git clone https://github.com/deepseek-ai/3FS.git
  cd 3FS || exit 1
  # https://github.com/deepseek-ai/3FS/commit/f6395e7d348b1383c168acaf2587fe7e0424dcf8
  # temporary avoid std::bit_cast not found
  git checkout f6395e7d348b1383c168acaf2587fe7e0424dcf8
  git submodule update --init --recursive

  ./patches/apply.sh
  cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSHUFFLE_METHOD=g++11
  cmake --build build -j 8 -t hf3fs_api_shared
popd || exit 1


pushd /tmp/hf3fs-rpm-build/ || exit 1
  rpmbuild --define "_topdir /tmp/hf3fs-rpm-build/" -bb /tmp/hf3fs-rpm-build/SPECS/hf3fs-usrbio.spec
popd || exit 1