#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class StandardUriTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(StandardUriTest, TestSimple) {
    {
        std::string redis_uri_str = "redis://user:pw@127.0.0.1:6379/abc/def?retry_count=2&timeout_ms=2000";
        StandardUri redis_uri = StandardUri::FromUri(redis_uri_str);
        ASSERT_EQ("redis", redis_uri.GetProtocol());
        ASSERT_EQ("user:pw", redis_uri.GetUserInfo());
        ASSERT_EQ("127.0.0.1", redis_uri.GetHostName());
        ASSERT_EQ(6379, redis_uri.GetPort());
        ASSERT_EQ("/abc/def", redis_uri.GetPath());
        ASSERT_EQ("2", redis_uri.GetParam("retry_count"));
        ASSERT_EQ("2000", redis_uri.GetParam("timeout_ms"));
    }
    {
        std::string redis_uri_str = "redis://127.0.0.1:6379/";
        StandardUri redis_uri = StandardUri::FromUri(redis_uri_str);
        ASSERT_EQ("redis", redis_uri.GetProtocol());
        ASSERT_EQ("", redis_uri.GetUserInfo());
        ASSERT_EQ("127.0.0.1", redis_uri.GetHostName());
        ASSERT_EQ(6379, redis_uri.GetPort());
        ASSERT_EQ("/", redis_uri.GetPath());
        ASSERT_EQ("", redis_uri.GetParam("retry_count"));
        ASSERT_EQ("", redis_uri.GetParam("timeout_ms"));
    }
    {
        std::string local_uri_str = "file:///tmp/kvcm";
        StandardUri local_uri = StandardUri::FromUri(local_uri_str);
        ASSERT_EQ("file", local_uri.GetProtocol());
        ASSERT_EQ("", local_uri.GetUserInfo());
        ASSERT_EQ("", local_uri.GetHostName());
        ASSERT_EQ(0, local_uri.GetPort());
        ASSERT_EQ("/tmp/kvcm", local_uri.GetPath());
    }
}

TEST_F(StandardUriTest, TestFileUri) {
    {
        // 带完整路径和查询参数的file URI，路径应包含文件名
        std::string uri = "file://nfs_01/tmp/test1/test.txt?offset=1&length=2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("nfs_01", data_storage_uri.GetHostName());
        ASSERT_EQ("/tmp/test1/test.txt", data_storage_uri.GetPath());
        ASSERT_EQ("1", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("2", data_storage_uri.GetParam("length"));

        data_storage_uri.SetParam("new_param", "v1");
        auto out_uri = data_storage_uri.ToUriString();
        StandardUri reparsed = StandardUri::FromUri(out_uri);
        ASSERT_TRUE(reparsed.Valid());
        ASSERT_EQ("nfs_01", reparsed.GetHostName());
        ASSERT_EQ("/tmp/test1/test.txt", reparsed.GetPath());
        ASSERT_EQ("1", reparsed.GetParam("offset"));
        ASSERT_EQ("2", reparsed.GetParam("length"));
        ASSERT_EQ("v1", reparsed.GetParam("new_param"));
    }

    {
        // 空host且无路径，仅有参数部分
        std::string uri = "file://?offset=1&length=2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("", data_storage_uri.GetHostName());
        ASSERT_EQ("", data_storage_uri.GetPath());
        ASSERT_EQ("1", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("2", data_storage_uri.GetParam("length"));

        auto out_uri = data_storage_uri.ToUriString();
        StandardUri reparsed = StandardUri::FromUri(out_uri);
        ASSERT_TRUE(reparsed.Valid());
        ASSERT_EQ("", reparsed.GetHostName());
        ASSERT_EQ("", reparsed.GetPath());
        ASSERT_EQ("1", reparsed.GetParam("offset"));
        ASSERT_EQ("2", reparsed.GetParam("length"));
    }

    {
        // 空host，根路径为"/"，带参数
        std::string uri = "file:///?offset=1&length=2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("", data_storage_uri.GetHostName());
        ASSERT_EQ("/", data_storage_uri.GetPath());
        ASSERT_EQ("1", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("2", data_storage_uri.GetParam("length"));

        auto out_uri = data_storage_uri.ToUriString();
        StandardUri reparsed = StandardUri::FromUri(out_uri);
        ASSERT_TRUE(reparsed.Valid());
        ASSERT_EQ("", reparsed.GetHostName());
        ASSERT_EQ("/", reparsed.GetPath());
        ASSERT_EQ("1", reparsed.GetParam("offset"));
        ASSERT_EQ("2", reparsed.GetParam("length"));
    }

    {
        // 仅根路径且无参数
        std::string uri = "file:///";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("", data_storage_uri.GetHostName());
        ASSERT_EQ("/", data_storage_uri.GetPath());
        ASSERT_EQ("", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("", data_storage_uri.GetParam("length"));
    }

    {
        // 空host，无参数，带路径和文件名
        std::string uri = "file:///tmp/test.txt";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("", data_storage_uri.GetHostName());
        ASSERT_EQ("/tmp/test.txt", data_storage_uri.GetPath());
        ASSERT_EQ("", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("", data_storage_uri.GetParam("length"));

        auto out_uri = data_storage_uri.ToUriString();
        StandardUri reparsed = StandardUri::FromUri(out_uri);
        ASSERT_TRUE(reparsed.Valid());
        ASSERT_EQ("", reparsed.GetHostName());
        ASSERT_EQ("/tmp/test.txt", reparsed.GetPath());
        ASSERT_EQ("", reparsed.GetParam("offset"));
        ASSERT_EQ("", reparsed.GetParam("length"));
    }

    {
        // 有host，无参数，有路径和文件名
        std::string uri = "file://worker01/tmp/test.txt";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("worker01", data_storage_uri.GetHostName());
        ASSERT_EQ("/tmp/test.txt", data_storage_uri.GetPath());
        ASSERT_EQ("", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("", data_storage_uri.GetParam("length"));
    }

    {
        // 有host，带路径和参数
        std::string uri = "file://worker01/tmp/test.txt?offset=1&length=2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("worker01", data_storage_uri.GetHostName());
        ASSERT_EQ("/tmp/test.txt", data_storage_uri.GetPath());
        ASSERT_EQ("1", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("2", data_storage_uri.GetParam("length"));
    }

    {
        // 有host，无路径，有参数
        std::string uri = "file://worker01?offset=1&length=2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("worker01", data_storage_uri.GetHostName());
        ASSERT_EQ("", data_storage_uri.GetPath());
        ASSERT_EQ("1", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("2", data_storage_uri.GetParam("length"));
    }

    {
        // 空host，无路径，有参数
        std::string uri = "file://?offset=1&length=2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("", data_storage_uri.GetHostName());
        ASSERT_EQ("", data_storage_uri.GetPath());
        ASSERT_EQ("1", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("2", data_storage_uri.GetParam("length"));
    }

    {
        // 空host，路径为"//"，无参数
        std::string uri = "file://///";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("", data_storage_uri.GetHostName());
        ASSERT_EQ("///", data_storage_uri.GetPath());
        ASSERT_EQ("", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("", data_storage_uri.GetParam("length"));
    }

    {
        // 空host，路径为"///"，带参数，参数值验证不对应应是对应的值
        std::string uri = "file://///?offset=100&length=2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("", data_storage_uri.GetHostName());
        ASSERT_EQ("///", data_storage_uri.GetPath());
        ASSERT_EQ("100", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("2", data_storage_uri.GetParam("length"));
    }

    // 非法协议格式，缺少 "://"
    {
        std::string uri = "file/tmp/test.txt";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_FALSE(data_storage_uri.Valid());
    }

    // 纯协议部分无host/path/query
    {
        std::string uri = "file://";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("", data_storage_uri.GetHostName());
        ASSERT_EQ("", data_storage_uri.GetPath());
    }

    // 有host但末尾带斜杠，无路径
    {
        std::string uri = "file://worker01/";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("worker01", data_storage_uri.GetHostName());
        ASSERT_EQ("/", data_storage_uri.GetPath());
    }

    // query只有键，没有值
    {
        std::string uri = "file://nfs_01/tmp/test.txt?offset&length=";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("nfs_01", data_storage_uri.GetHostName());
        ASSERT_EQ("/tmp/test.txt", data_storage_uri.GetPath());
        ASSERT_EQ("", data_storage_uri.GetParam("offset"));
        ASSERT_EQ("", data_storage_uri.GetParam("length"));
    }

    // 测试 ToUriString 反解回来的字符串包含所有部分
    {
        std::string uri = "file://host/tmp/path/file.txt?param1=val1&param2=val2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        ASSERT_EQ("host", data_storage_uri.GetHostName());
        ASSERT_EQ("/tmp/path/file.txt", data_storage_uri.GetPath());

        data_storage_uri.SetParam("newparam", "newval");
        std::string new_uri = data_storage_uri.ToUriString();
        // 检查拼接字符串包含新增参数和已有参数
        ASSERT_NE(std::string::npos, new_uri.find("newparam=newval"));
        ASSERT_NE(std::string::npos, new_uri.find("param1=val1"));
        ASSERT_NE(std::string::npos, new_uri.find("param2=val2"));
    }

    // 测试无参数时 ToUriString 不添加 '?'
    {
        std::string uri = "file://host/tmp/file";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        std::string out_uri = data_storage_uri.ToUriString();
        ASSERT_EQ(out_uri.find('?'), std::string::npos);
    }

    // 多次设置和获取参数
    {
        StandardUri d;
        d.SetProtocol("file");
        d.SetHostName("nfs_01");
        d.SetPath("/tmp/test.txt");
        d.SetParam("key1", "value1");
        d.SetParam("key2", "value2");
        ASSERT_EQ("value1", d.GetParam("key1"));
        ASSERT_EQ("value2", d.GetParam("key2"));
        ASSERT_EQ("", d.GetParam("nosuchkey"));
        d.SetParam("key1", "newvalue");
        ASSERT_EQ("newvalue", d.GetParam("key1"));

        auto out_uri = d.ToUriString();
        ASSERT_EQ("file://nfs_01/tmp/test.txt?key1=newvalue&key2=value2", out_uri);

        d.SetProtocol("");
        out_uri = d.ToUriString();
        ASSERT_EQ("", out_uri);

        d.SetProtocol("file");
        d.SetPath("");
        out_uri = d.ToUriString();
        ASSERT_EQ("file://nfs_01?key1=newvalue&key2=value2", out_uri);
    }

    // 参数值包含特殊字符，需要正确编码（测试ToUriString不会破坏参数结构）
    {
        std::string uri = "file://host/tmp/file.txt?param=hello world&another=1&2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        std::string out_uri = data_storage_uri.ToUriString();

        // 这里假设不自动编码，只验证原始字符串中包含参数键和值
        ASSERT_NE(out_uri.find("param=hello world"), std::string::npos);
        ASSERT_NE(out_uri.find("another=1"), std::string::npos);
    }

    // 空path，有参数，ToUriString拼接验证
    {
        std::string uri = "file://host?param1=val1";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        std::string out_uri = data_storage_uri.ToUriString();
        ASSERT_NE(out_uri.find("file://host?param1=val1"), std::string::npos);
    }

    // 多个参数中含空值和无值参数，ToUriString拼接正确性
    {
        std::string uri = "file://host/tmp/file?param1=&param2&param3=val3";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());

        std::string out_uri = data_storage_uri.ToUriString();
        ASSERT_NE(out_uri.find("param1="), std::string::npos);
        ASSERT_NE(out_uri.find("param2="), std::string::npos); // param2无值同样会以 param2= 拼出
        ASSERT_NE(out_uri.find("param3=val3"), std::string::npos);
    }

    // 参数值包含特殊字符，ToUriString拼接后再解析，验证参数完整性
    {
        std::string uri = "file://host/tmp/file.txt?param=hello world&another=1&2";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        std::string out_uri = data_storage_uri.ToUriString();

        // 重新解析拼接结果
        StandardUri reparsed = StandardUri::FromUri(out_uri);
        ASSERT_TRUE(reparsed.Valid());
        ASSERT_EQ("host", reparsed.GetHostName());
        ASSERT_EQ("/tmp/file.txt", reparsed.GetPath());
        ASSERT_EQ("hello world", reparsed.GetParam("param"));
        ASSERT_EQ("1", reparsed.GetParam("another"));
        ASSERT_EQ("", reparsed.GetParam("2"));
    }

    // 空path，有参数，ToUriString拼接再解析校验
    {
        std::string uri = "file://host?param1=val1";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());
        std::string out_uri = data_storage_uri.ToUriString();

        StandardUri reparsed = StandardUri::FromUri(out_uri);
        ASSERT_TRUE(reparsed.Valid());
        ASSERT_EQ("host", reparsed.GetHostName());
        ASSERT_EQ("", reparsed.GetPath());
        ASSERT_EQ("val1", reparsed.GetParam("param1"));
    }

    // 多参数含空值和无值，ToUriString拼接后再解析，参数完整性校验
    {
        std::string uri = "file://host/tmp/file?param1=&param2&param3=val3";
        StandardUri data_storage_uri = StandardUri::FromUri(uri);
        ASSERT_TRUE(data_storage_uri.Valid());

        std::string out_uri = data_storage_uri.ToUriString();

        StandardUri reparsed = StandardUri::FromUri(out_uri);
        ASSERT_TRUE(reparsed.Valid());
        ASSERT_EQ("host", reparsed.GetHostName());
        ASSERT_EQ("/tmp/file", reparsed.GetPath());
        ASSERT_EQ("", reparsed.GetParam("param1"));
        ASSERT_EQ("", reparsed.GetParam("param2")); // 无值参数解析为空字符串
        ASSERT_EQ("val3", reparsed.GetParam("param3"));
    }
}

TEST_F(StandardUriTest, TestInvalidPort) {
    {
        std::string redis_uri_str = "redis://user:pw@127.0.0.1";
        StandardUri redis_uri = StandardUri::FromUri(redis_uri_str);
        ASSERT_EQ("redis", redis_uri.GetProtocol());
        ASSERT_EQ("user:pw", redis_uri.GetUserInfo());
        ASSERT_EQ("127.0.0.1", redis_uri.GetHostName());
        ASSERT_EQ(0, redis_uri.GetPort()); // default 0
    }
    {
        std::string redis_uri_str = "redis://user:pw@127.0.0.1:abcd/";
        StandardUri redis_uri = StandardUri::FromUri(redis_uri_str);
        ASSERT_FALSE(redis_uri.Valid());
    }
}
} // namespace kv_cache_manager
