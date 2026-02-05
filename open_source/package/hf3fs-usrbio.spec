Name:           libhf3fs_api_shared
Version:        1.2.1
Release:        1%{?dist}
Summary:        hf3fs shared library and headers

License:        MIT
URL:            https://github.com/deepseek-ai/3FS/

%description
This package provides the hf3fs API shared libraries and header files.

%prep
# 不需要 %setup

%build
# 不需要编译

%install
rm -rf $RPM_BUILD_ROOT
mkdir -p $RPM_BUILD_ROOT/usr/include
mkdir -p $RPM_BUILD_ROOT/usr/lib64

install -m 644 /tmp/hf3fs-dependencies/3FS/src/lib/api/hf3fs_usrbio.h \
    $RPM_BUILD_ROOT/usr/include/

install -m 755 /tmp/hf3fs-dependencies/3FS/build/src/lib/api/libhf3fs_api_shared.so \
    $RPM_BUILD_ROOT/usr/lib64/

install -m 755 /usr/lib/*-linux-gnu/libboost_context.so.1.71.0 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libboost_filesystem.so.1.71.0 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libboost_program_options.so.1.71.0 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libboost_regex.so.1.71.0 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libboost_system.so.1.71.0 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libboost_thread.so.1.71.0 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libboost_atomic.so.1.71.0 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libdouble-conversion.so.3 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libgflags.so.2.2 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libglog.so.0 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libevent-2.1.so.7 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libdwarf.so.1 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libicui18n.so.66 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libicuuc.so.66 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libicudata.so.66 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libunwind.so.8 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libssl.so.1.1 \
    $RPM_BUILD_ROOT/usr/lib64/
install -m 755 /usr/lib/*-linux-gnu/libcrypto.so.1.1 \
    $RPM_BUILD_ROOT/usr/lib64/

%files
%defattr(-,root,root,-)
/usr/include/hf3fs_usrbio.h
/usr/lib64/libhf3fs_api_shared.so
/usr/lib64/libboost_context.so.1.71.0
/usr/lib64/libboost_filesystem.so.1.71.0
/usr/lib64/libboost_program_options.so.1.71.0
/usr/lib64/libboost_regex.so.1.71.0
/usr/lib64/libboost_system.so.1.71.0
/usr/lib64/libboost_thread.so.1.71.0
/usr/lib64/libboost_atomic.so.1.71.0
/usr/lib64/libdouble-conversion.so.3
/usr/lib64/libgflags.so.2.2
/usr/lib64/libglog.so.0
/usr/lib64/libevent-2.1.so.7
/usr/lib64/libdwarf.so.1
/usr/lib64/libicui18n.so.66
/usr/lib64/libicuuc.so.66
/usr/lib64/libicudata.so.66
/usr/lib64/libunwind.so.8
/usr/lib64/libssl.so.1.1
/usr/lib64/libcrypto.so.1.1

%post
/sbin/ldconfig

%postun
/sbin/ldconfig

%changelog
* Thu Feb 05 2026 xiyu.wxy <xiyu.wxy@alibaba-inc.com> - 1.0.1-1
* Fri May 23 2025 LXQ <lxq271332@alibaba-inc.com> - 1.0.0-1
- Initial release