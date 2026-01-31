#include <gtest/gtest.h>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class JsonizableTest : public TESTBASE {
public:
};

enum class EnumClass {
    EC_1 = 1,
    EC_2 = 2,
    EC_3 = 3,
};

class JsonClass : public Jsonizable {
public:
    JsonClass() = default;

    ~JsonClass() override {
        if (raw_json_ptr_) {
            delete raw_json_ptr_;
        }
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "int_value", int_value_);
        Put(writer, "bool_value", bool_value_);
        Put(writer, "int8_t_value", int8_t_value_);
        Put(writer, "uint8_t_value", uint8_t_value_);
        Put(writer, "int16_t_value", int16_t_value_);
        Put(writer, "uint16_t_value", uint16_t_value_);
        Put(writer, "int32_t_value", int32_t_value_);
        Put(writer, "uint32_t_value", uint32_t_value_);
        Put(writer, "int64_t_value", int64_t_value_);
        Put(writer, "uint64_t_value", uint64_t_value_);
        Put(writer, "double_value", double_value_);
        Put(writer, "float_value", float_value_);
        Put(writer, "string_value", string_value_);
        Put(writer, "int_values", int_values_);
        Put(writer, "map_si", map_si_);
        Put(writer, "map_ss", map_ss_);
        Put(writer, "inner_json", inner_json_);
        Put(writer, "inner_jsons", inner_jsons_);
        Put(writer, "inner_json_map", inner_json_map_);
        Put(writer, "enum_value", enum_value_);
        Put(writer, "raw_json_ptr", raw_json_ptr_);
        Put(writer, "json_ptr", json_ptr_);
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "int_value", int_value_, 5);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "bool_value", bool_value_, true);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "int8_t_value", int8_t_value_, (int8_t)11);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "uint8_t_value", uint8_t_value_, (uint8_t)22);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "int16_t_value", int16_t_value_, (int16_t)33);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "uint16_t_value", uint16_t_value_, (uint16_t)44);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "int32_t_value", int32_t_value_, (int32_t)55);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "uint32_t_value", uint32_t_value_, (uint32_t)66);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "int64_t_value", int64_t_value_, (int64_t)77);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "uint64_t_value", uint64_t_value_, (uint64_t)88);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "double_value", double_value_, 111.111);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "float_value", float_value_, (float)222.222);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "string_value", string_value_, std::string("333.333"));
        KVCM_JSON_GET_MACRO(rapid_value, "int_values", int_values_);
        KVCM_JSON_GET_MACRO(rapid_value, "map_si", map_si_);
        KVCM_JSON_GET_MACRO(rapid_value, "map_ss", map_ss_);
        KVCM_JSON_GET_MACRO(rapid_value, "inner_json", inner_json_);
        KVCM_JSON_GET_MACRO(rapid_value, "inner_jsons", inner_jsons_);
        KVCM_JSON_GET_MACRO(rapid_value, "inner_json_map", inner_json_map_);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "enum_value", enum_value_, EnumClass::EC_1);
        KVCM_JSON_GET_MACRO(rapid_value, "raw_json_ptr", raw_json_ptr_);
        KVCM_JSON_GET_MACRO(rapid_value, "json_ptr", json_ptr_);
        return true;
    }

    class InnerJsonClass : public Jsonizable {
    public:
        InnerJsonClass() = default;

        InnerJsonClass(int int_value, const std::string &string_value, const std::vector<int> &int_values)
            : inner_int_value_(int_value), inner_string_value_(string_value), inner_int_values_(int_values) {}

        ~InnerJsonClass() override = default;
        void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
            Put(writer, "inner_int_value", inner_int_value_);
            Put(writer, "inner_string_value", inner_string_value_);
            Put(writer, "inner_int_values", inner_int_values_);
        }

        bool FromRapidValue(const rapidjson::Value &rapid_value) override {
            KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "inner_int_value", inner_int_value_, 1001);
            KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "inner_string_value", inner_string_value_, std::string("1002"));
            KVCM_JSON_GET_MACRO(rapid_value, "inner_int_values", inner_int_values_);
            return true;
        }

        bool operator==(const InnerJsonClass &other) const {
            return inner_int_value_ == other.inner_int_value_ && inner_string_value_ == other.inner_string_value_ &&
                   inner_int_values_ == other.inner_int_values_;
        }

    private:
        int inner_int_value_ = 9999;
        std::string inner_string_value_;
        std::vector<int> inner_int_values_;
    };

private:
    // simple value
    int int_value_;
    bool bool_value_;
    int8_t int8_t_value_;
    uint8_t uint8_t_value_;
    int16_t int16_t_value_;
    uint16_t uint16_t_value_;
    int32_t int32_t_value_;
    uint32_t uint32_t_value_;
    int64_t int64_t_value_;
    uint64_t uint64_t_value_;
    double double_value_;
    float float_value_;
    std::string string_value_;
    // conatiner
    std::vector<int> int_values_;
    std::map<std::string, int> map_si_;
    std::map<std::string, std::string> map_ss_;
    // inner json
    InnerJsonClass inner_json_;
    std::vector<InnerJsonClass> inner_jsons_;
    std::map<std::string, InnerJsonClass> inner_json_map_;
    // enum
    EnumClass enum_value_;
    // pointer
    InnerJsonClass *raw_json_ptr_ = nullptr;
    std::shared_ptr<InnerJsonClass> json_ptr_;
};

TEST_F(JsonizableTest, TestParseError) {
    std::string json_str = R"(
        {
            "bool_value" : false,
            "int8_t_value" : 15,
            "int64_t_value" : 5555,
    )";
    JsonClass json_objct;
    ASSERT_FALSE(json_objct.FromJsonString(json_str));
}

TEST_F(JsonizableTest, TestFromJsonErrorType) {
    std::string json_str = R"(
        {
            "int8_t_value" : "15"
        }
    )";
    JsonClass json_objct;
    ASSERT_FALSE(json_objct.FromJsonString(json_str));
}

TEST_F(JsonizableTest, TestFromJsonNull) {
    std::string json_str = R"(
        {
            "int8_t_value" : null
        }
    )";
    JsonClass json_objct;
    ASSERT_FALSE(json_objct.FromJsonString(json_str));
}

TEST_F(JsonizableTest, TestSimple) {
    std::string expected =
        "{\"int_value\":5,\"bool_value\":false,\"int8_t_value\":15,\"uint8_t_value\":22,\"int16_t_value\":33,\"uint16_"
        "t_"
        "value\":44,\"int32_t_value\":55,\"uint32_t_value\":66,\"int64_t_value\":5555,\"uint64_t_value\":88,\"double_"
        "value\":111.111,\"float_value\":222.2220001220703,\"string_value\":\"333.333\",\"int_values\":[],\"map_si\":{}"
        ",\"map_ss\":{},\"inner_json\":{\"inner_int_value\":9999,\"inner_string_value\":\"\",\"inner_int_values\":[]},"
        "\"inner_jsons\":[],\"inner_json_map\":{},\"enum_value\":1,\"raw_json_ptr\":null,\"json_ptr\":null}";
    {
        std::string json_str = R"(
            {
                "bool_value" : false,
                "int8_t_value" : 15,
                "int64_t_value" : 5555
            }
        )";
        JsonClass json_objct;
        ASSERT_TRUE(json_objct.FromJsonString(json_str));
        ASSERT_EQ(5, json_objct.int_value_);
        ASSERT_EQ(false, json_objct.bool_value_);
        ASSERT_EQ((int8_t)15, json_objct.int8_t_value_);
        ASSERT_EQ((uint8_t)22, json_objct.uint8_t_value_);
        ASSERT_EQ((int16_t)33, json_objct.int16_t_value_);
        ASSERT_EQ((uint16_t)44, json_objct.uint16_t_value_);
        ASSERT_EQ((int32_t)55, json_objct.int32_t_value_);
        ASSERT_EQ((uint32_t)66, json_objct.uint32_t_value_);
        ASSERT_EQ((int64_t)5555, json_objct.int64_t_value_);
        ASSERT_EQ((uint64_t)88, json_objct.uint64_t_value_);
        ASSERT_EQ((double)111.111, json_objct.double_value_);
        ASSERT_EQ((float)222.222, json_objct.float_value_);
        ASSERT_EQ(std::string("333.333"), json_objct.string_value_);
        ASSERT_EQ(EnumClass::EC_1, json_objct.enum_value_);
        ASSERT_EQ(nullptr, json_objct.raw_json_ptr_);
        ASSERT_EQ(nullptr, json_objct.json_ptr_);
        ASSERT_EQ(expected, json_objct.ToJsonString());
    }
    {
        std::string json_str = R"(
            {
                "bool_value" : false,
                "int8_t_value" : 15,
                "int64_t_value" : 5555,
                "not_exist" : "useless"
            }
        )";
        JsonClass json_objct;
        ASSERT_TRUE(json_objct.FromJsonString(json_str));
        ASSERT_EQ(5, json_objct.int_value_);
        ASSERT_EQ(false, json_objct.bool_value_);
        ASSERT_EQ((int8_t)15, json_objct.int8_t_value_);
        ASSERT_EQ((uint8_t)22, json_objct.uint8_t_value_);
        ASSERT_EQ((int16_t)33, json_objct.int16_t_value_);
        ASSERT_EQ((uint16_t)44, json_objct.uint16_t_value_);
        ASSERT_EQ((int32_t)55, json_objct.int32_t_value_);
        ASSERT_EQ((uint32_t)66, json_objct.uint32_t_value_);
        ASSERT_EQ((int64_t)5555, json_objct.int64_t_value_);
        ASSERT_EQ((uint64_t)88, json_objct.uint64_t_value_);
        ASSERT_EQ((double)111.111, json_objct.double_value_);
        ASSERT_EQ((float)222.222, json_objct.float_value_);
        ASSERT_EQ(std::string("333.333"), json_objct.string_value_);
        ASSERT_EQ(expected, json_objct.ToJsonString());
    }
}

TEST_F(JsonizableTest, TestContainer) {
    std::string json_str = R"(
        {
            "bool_value" : false,
            "int8_t_value" : 15,
            "int64_t_value" : 5555,
            "int_values" : [1, 2, 3],
            "map_si" : {
                "k1" : 1,
                "k2" : 2
            },
            "map_ss" : {
                "k1" : "v1",
                "k2" : "v2"
            }
        }
    )";
    JsonClass json_objct;
    ASSERT_TRUE(json_objct.FromJsonString(json_str));
    ASSERT_EQ(5, json_objct.int_value_);
    ASSERT_EQ(false, json_objct.bool_value_);
    ASSERT_EQ((int8_t)15, json_objct.int8_t_value_);
    ASSERT_EQ((int64_t)5555, json_objct.int64_t_value_);
    ASSERT_EQ((std::vector<int>{1, 2, 3}), json_objct.int_values_);
    ASSERT_EQ((std::map<std::string, int>{{"k1", 1}, {"k2", 2}}), json_objct.map_si_);
    ASSERT_EQ((std::map<std::string, std::string>{{"k1", "v1"}, {"k2", "v2"}}), json_objct.map_ss_);
    ASSERT_EQ(
        (std::string("{\"int_value\":5,\"bool_value\":false,\"int8_t_value\":15,\"uint8_t_value\":22,\"int16_t_value\":"
                     "33,\"uint16_t_value\":44,\"int32_t_value\":55,\"uint32_t_value\":66,\"int64_t_value\":5555,"
                     "\"uint64_t_value\":88,\"double_value\":111.111,\"float_value\":222.2220001220703,\"string_"
                     "value\":\"333.333\",\"int_values\":[1,2,3],\"map_si\":{\"k1\":1,\"k2\":2},\"map_ss\":{\"k1\":"
                     "\"v1\",\"k2\":\"v2\"},\"inner_json\":{\"inner_int_value\":9999,\"inner_string_value\":\"\","
                     "\"inner_int_values\":[]},\"inner_jsons\":[],\"inner_json_map\":{},\"enum_value\":1,\"raw_json_"
                     "ptr\":null,\"json_ptr\":null}")),
        json_objct.ToJsonString());
}

TEST_F(JsonizableTest, TestInnerJson) {
    std::string json_str = R"(
        {
            "bool_value" : false,
            "int8_t_value" : 15,
            "int64_t_value" : 5555,
            "int_values" : [1, 2, 3],
            "map_si" : {
                "k1" : 1,
                "k2" : 2
            },
            "map_ss" : {
                "k1" : "v1",
                "k2" : "v2"
            },
            "inner_json" : {
                "inner_int_value" : 2001,
                "inner_string_value" : "hello Mercury",
                "inner_int_values" : [
                    4, 5, 6
                ]
            },
            "inner_jsons" : [
                {
                    "inner_int_value" : 3001,
                    "inner_string_value" : "hello Venus",
                    "inner_int_values" : [
                        7, 8, 9
                    ]
                },
                {
                    "inner_int_value" : 3002,
                    "inner_string_value" : "hello Earth",
                    "inner_int_values" : [
                        10, 11, 12
                    ]
                }
            ],
            "inner_json_map" : {
                "boy" : {
                    "inner_int_value" : 4001,
                    "inner_string_value" : "hello Mars",
                    "inner_int_values" : [
                        13, 14, 15
                    ]
                },
                "girl" : {
                    "inner_int_value" : 4002,
                    "inner_string_value" : "hello Jupiter",
                    "inner_int_values" : [
                        16, 17, 18
                    ]
                }
            }
        }
    )";
    JsonClass json_objct;
    ASSERT_TRUE(json_objct.FromJsonString(json_str));
    ASSERT_EQ(5, json_objct.int_value_);
    ASSERT_EQ(false, json_objct.bool_value_);
    ASSERT_EQ((int8_t)15, json_objct.int8_t_value_);
    ASSERT_EQ((int64_t)5555, json_objct.int64_t_value_);
    ASSERT_EQ((std::vector<int>{1, 2, 3}), json_objct.int_values_);
    ASSERT_EQ((std::map<std::string, int>{{"k1", 1}, {"k2", 2}}), json_objct.map_si_);
    ASSERT_EQ((std::map<std::string, std::string>{{"k1", "v1"}, {"k2", "v2"}}), json_objct.map_ss_);
    ASSERT_EQ(JsonClass::InnerJsonClass(2001, "hello Mercury", {4, 5, 6}), json_objct.inner_json_);
    ASSERT_EQ((std::vector<JsonClass::InnerJsonClass>{JsonClass::InnerJsonClass(3001, "hello Venus", {7, 8, 9}),
                                                      JsonClass::InnerJsonClass(3002, "hello Earth", {10, 11, 12})}),
              json_objct.inner_jsons_);
    ASSERT_EQ((std::map<std::string, JsonClass::InnerJsonClass>{
                  {"boy", JsonClass::InnerJsonClass(4001, "hello Mars", {13, 14, 15})},
                  {"girl", JsonClass::InnerJsonClass(4002, "hello Jupiter", {16, 17, 18})},
              }),
              json_objct.inner_json_map_);
    ASSERT_EQ(
        (std::string(
            "{\"int_value\":5,\"bool_value\":false,\"int8_t_value\":15,\"uint8_t_value\":22,\"int16_t_value\":33,"
            "\"uint16_t_value\":44,\"int32_t_value\":55,\"uint32_t_value\":66,\"int64_t_value\":5555,\"uint64_t_"
            "value\":88,\"double_value\":111.111,\"float_value\":222.2220001220703,\"string_value\":\"333.333\",\"int_"
            "values\":[1,2,3],\"map_si\":{\"k1\":1,\"k2\":2},\"map_ss\":{\"k1\":\"v1\",\"k2\":\"v2\"},\"inner_json\":{"
            "\"inner_int_value\":2001,\"inner_string_value\":\"hello "
            "Mercury\",\"inner_int_values\":[4,5,6]},\"inner_jsons\":[{\"inner_int_value\":3001,\"inner_string_value\":"
            "\"hello Venus\",\"inner_int_values\":[7,8,9]},{\"inner_int_value\":3002,\"inner_string_value\":\"hello "
            "Earth\",\"inner_int_values\":[10,11,12]}],\"inner_json_map\":{\"boy\":{\"inner_int_value\":4001,\"inner_"
            "string_value\":\"hello "
            "Mars\",\"inner_int_values\":[13,14,15]},\"girl\":{\"inner_int_value\":4002,\"inner_string_value\":\"hello "
            "Jupiter\",\"inner_int_values\":[16,17,18]}},\"enum_value\":1,\"raw_json_ptr\":null,\"json_ptr\":null}")),
        json_objct.ToJsonString());
}

TEST_F(JsonizableTest, TestEnumNotEq) {
    {
        std::string json_str = R"(
            {
                "enum_value" : 4
            }
        )";
        JsonClass json_objct;
        ASSERT_TRUE(json_objct.FromJsonString(json_str));
        ASSERT_EQ(4, static_cast<int>(json_objct.enum_value_));
        ASSERT_FALSE(EnumClass::EC_1 == json_objct.enum_value_);
        ASSERT_FALSE(EnumClass::EC_2 == json_objct.enum_value_);
        ASSERT_FALSE(EnumClass::EC_3 == json_objct.enum_value_);
    }
}

TEST_F(JsonizableTest, TestPointer) {
    std::string json_str = R"(
        {
            "bool_value" : false,
            "int8_t_value" : 15,
            "int64_t_value" : 5555,
            "int_values" : [1, 2, 3],
            "map_si" : {
                "k1" : 1,
                "k2" : 2
            },
            "map_ss" : {
                "k1" : "v1",
                "k2" : "v2"
            },
            "inner_json" : {
                "inner_int_value" : 2001,
                "inner_string_value" : "hello Mercury",
                "inner_int_values" : [
                    4, 5, 6
                ]
            },
            "inner_jsons" : [
                {
                    "inner_int_value" : 3001,
                    "inner_string_value" : "hello Venus",
                    "inner_int_values" : [
                        7, 8, 9
                    ]
                },
                {
                    "inner_int_value" : 3002,
                    "inner_string_value" : "hello Earth",
                    "inner_int_values" : [
                        10, 11, 12
                    ]
                }
            ],
            "inner_json_map" : {
                "boy" : {
                    "inner_int_value" : 4001,
                    "inner_string_value" : "hello Mars",
                    "inner_int_values" : [
                        13, 14, 15
                    ]
                },
                "girl" : {
                    "inner_int_value" : 4002,
                    "inner_string_value" : "hello Jupiter",
                    "inner_int_values" : [
                        16, 17, 18
                    ]
                }
            },
            "raw_json_ptr" : {
                "inner_int_value" : 5002,
                "inner_string_value" : "hello Saturn",
                "inner_int_values" : [
                    19, 20, 21
                ]
            },
            "json_ptr" : {
                "inner_int_value" : 6002,
                "inner_string_value" : "hello Uranus",
                "inner_int_values" : [
                    22, 23, 24
                ]
            }
        }
    )";
    JsonClass json_objct;
    ASSERT_TRUE(json_objct.FromJsonString(json_str));
    ASSERT_EQ(5, json_objct.int_value_);
    ASSERT_EQ(false, json_objct.bool_value_);
    ASSERT_EQ((int8_t)15, json_objct.int8_t_value_);
    ASSERT_EQ((int64_t)5555, json_objct.int64_t_value_);
    ASSERT_EQ((std::vector<int>{1, 2, 3}), json_objct.int_values_);
    ASSERT_EQ((std::map<std::string, int>{{"k1", 1}, {"k2", 2}}), json_objct.map_si_);
    ASSERT_EQ((std::map<std::string, std::string>{{"k1", "v1"}, {"k2", "v2"}}), json_objct.map_ss_);
    ASSERT_EQ(JsonClass::InnerJsonClass(2001, "hello Mercury", {4, 5, 6}), json_objct.inner_json_);
    ASSERT_EQ((std::vector<JsonClass::InnerJsonClass>{JsonClass::InnerJsonClass(3001, "hello Venus", {7, 8, 9}),
                                                      JsonClass::InnerJsonClass(3002, "hello Earth", {10, 11, 12})}),
              json_objct.inner_jsons_);
    ASSERT_EQ((std::map<std::string, JsonClass::InnerJsonClass>{
                  {"boy", JsonClass::InnerJsonClass(4001, "hello Mars", {13, 14, 15})},
                  {"girl", JsonClass::InnerJsonClass(4002, "hello Jupiter", {16, 17, 18})},
              }),
              json_objct.inner_json_map_);
    ASSERT_EQ(JsonClass::InnerJsonClass(5002, "hello Saturn", {19, 20, 21}), *(json_objct.raw_json_ptr_));
    ASSERT_EQ(JsonClass::InnerJsonClass(6002, "hello Uranus", {22, 23, 24}), *(json_objct.json_ptr_));
    ASSERT_EQ(
        (std::string(
            "{\"int_value\":5,\"bool_value\":false,\"int8_t_value\":15,\"uint8_t_value\":22,\"int16_t_value\":33,"
            "\"uint16_t_value\":44,\"int32_t_value\":55,\"uint32_t_value\":66,\"int64_t_value\":5555,\"uint64_t_"
            "value\":88,\"double_value\":111.111,\"float_value\":222.2220001220703,\"string_value\":\"333.333\",\"int_"
            "values\":[1,2,3],\"map_si\":{\"k1\":1,\"k2\":2},\"map_ss\":{\"k1\":\"v1\",\"k2\":\"v2\"},\"inner_json\":{"
            "\"inner_int_value\":2001,\"inner_string_value\":\"hello "
            "Mercury\",\"inner_int_values\":[4,5,6]},\"inner_jsons\":[{\"inner_int_value\":3001,\"inner_string_value\":"
            "\"hello Venus\",\"inner_int_values\":[7,8,9]},{\"inner_int_value\":3002,\"inner_string_value\":\"hello "
            "Earth\",\"inner_int_values\":[10,11,12]}],\"inner_json_map\":{\"boy\":{\"inner_int_value\":4001,\"inner_"
            "string_value\":\"hello "
            "Mars\",\"inner_int_values\":[13,14,15]},\"girl\":{\"inner_int_value\":4002,\"inner_string_value\":\"hello "
            "Jupiter\",\"inner_int_values\":[16,17,18]}},\"enum_value\":1,\"raw_json_ptr\":{\"inner_int_value\":5002,"
            "\"inner_string_value\":\"hello "
            "Saturn\",\"inner_int_values\":[19,20,21]},\"json_ptr\":{\"inner_int_value\":6002,\"inner_string_value\":"
            "\"hello Uranus\",\"inner_int_values\":[22,23,24]}}")),
        json_objct.ToJsonString());
}

TEST_F(JsonizableTest, TestToJsonString) {
    {
        std::map<std::string, int> m = {{"one", 1}, {"two", 2}, {"three", 3}};
        ASSERT_EQ(std::string("{\"one\":1,\"three\":3,\"two\":2}"), Jsonizable::ToJsonString(m));
    }
    {
        std::map<std::string, JsonClass::InnerJsonClass> m = {
            {"one", JsonClass::InnerJsonClass(1, "1", {11, 12, 13})},
            {"two", JsonClass::InnerJsonClass(2, "2", {21, 22, 23})},
            {"three", JsonClass::InnerJsonClass(3, "3", {31, 32, 33})},
        };
        ASSERT_EQ(std::string(
                      "{\"one\":{\"inner_int_value\":1,\"inner_string_value\":\"1\",\"inner_int_values\":[11,12,13]},"
                      "\"three\":{\"inner_int_value\":3,\"inner_string_value\":\"3\",\"inner_int_values\":[31,32,33]},"
                      "\"two\":{\"inner_int_value\":2,\"inner_string_value\":\"2\",\"inner_int_values\":[21,22,23]}}"),
                  Jsonizable::ToJsonString(m));
    }
    {
        std::vector<std::string> v = {"one", "two", "three"};
        ASSERT_EQ(std::string("[\"one\",\"two\",\"three\"]"), Jsonizable::ToJsonString(v));
    }
    {
        std::vector<JsonClass::InnerJsonClass> v = {
            JsonClass::InnerJsonClass(1, "1", {11, 12, 13}),
            JsonClass::InnerJsonClass(2, "2", {21, 22, 23}),
            JsonClass::InnerJsonClass(3, "3", {31, 32, 33}),
        };
        ASSERT_EQ(std::string("[{\"inner_int_value\":1,\"inner_string_value\":\"1\",\"inner_int_values\":[11,12,13]},{"
                              "\"inner_int_value\":2,\"inner_string_value\":\"2\",\"inner_int_values\":[21,22,23]},{"
                              "\"inner_int_value\":3,\"inner_string_value\":\"3\",\"inner_int_values\":[31,32,33]}]"),
                  Jsonizable::ToJsonString(v));
    }
}

TEST_F(JsonizableTest, TestFromJsonString) {
    {
        std::string json_str("{\"one\":1,\"three\":3,\"two\":2}");
        std::map<std::string, int> expect = {{"one", 1}, {"two", 2}, {"three", 3}};
        std::map<std::string, int> real;
        ASSERT_TRUE(Jsonizable::FromJsonString(json_str, real));
        ASSERT_EQ(expect, real);
    }
    {
        std::string json_str(
            "{\"one\":{\"inner_int_value\":1,\"inner_string_value\":\"1\",\"inner_int_values\":[11,12,13]},"
            "\"three\":{\"inner_int_value\":3,\"inner_string_value\":\"3\",\"inner_int_values\":[31,32,33]},"
            "\"two\":{\"inner_int_value\":2,\"inner_string_value\":\"2\",\"inner_int_values\":[21,22,23]}}");
        std::map<std::string, JsonClass::InnerJsonClass> expect = {
            {"one", JsonClass::InnerJsonClass(1, "1", {11, 12, 13})},
            {"two", JsonClass::InnerJsonClass(2, "2", {21, 22, 23})},
            {"three", JsonClass::InnerJsonClass(3, "3", {31, 32, 33})},
        };
        std::map<std::string, JsonClass::InnerJsonClass> real;
        ASSERT_TRUE(Jsonizable::FromJsonString(json_str, real));
        ASSERT_EQ(expect, real);
    }
    {
        std::string json_str("[\"one\",\"two\",\"three\"]");
        std::vector<std::string> expect = {"one", "two", "three"};
        std::vector<std::string> real;
        ASSERT_TRUE(Jsonizable::FromJsonString(json_str, real));
        ASSERT_EQ(expect, real);
    }
    {
        std::string json_str("[{\"inner_int_value\":1,\"inner_string_value\":\"1\",\"inner_int_values\":[11,12,13]},{"
                             "\"inner_int_value\":2,\"inner_string_value\":\"2\",\"inner_int_values\":[21,22,23]},{"
                             "\"inner_int_value\":3,\"inner_string_value\":\"3\",\"inner_int_values\":[31,32,33]}]");
        std::vector<JsonClass::InnerJsonClass> expect = {
            JsonClass::InnerJsonClass(1, "1", {11, 12, 13}),
            JsonClass::InnerJsonClass(2, "2", {21, 22, 23}),
            JsonClass::InnerJsonClass(3, "3", {31, 32, 33}),
        };
        std::vector<JsonClass::InnerJsonClass> real;
        ASSERT_TRUE(Jsonizable::FromJsonString(json_str, real));
        ASSERT_EQ(expect, real);
    }
}

TEST_F(JsonizableTest, TestNestedContainer) {
    class NestedContainerJsonClass : public Jsonizable {
    public:
        void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
            Put(writer, "vec_int_ptr", vec_int_ptr_);
            Put(writer, "vec_vec_int", vec_vec_int_);
            Put(writer, "map_str_vec_str", map_str_vec_str_);
            Put(writer, "map_str_map_str_str", map_str_map_str_str_);
        }

        bool FromRapidValue(const rapidjson::Value &rapid_value) override {
            KVCM_JSON_GET_MACRO(rapid_value, "vec_int_ptr", vec_int_ptr_);
            KVCM_JSON_GET_MACRO(rapid_value, "vec_vec_int", vec_vec_int_);
            KVCM_JSON_GET_MACRO(rapid_value, "map_str_vec_str", map_str_vec_str_);
            KVCM_JSON_GET_MACRO(rapid_value, "map_str_map_str_str", map_str_map_str_str_);
            return true;
        }

    private:
        std::shared_ptr<std::vector<int>> vec_int_ptr_;
        std::vector<std::vector<int>> vec_vec_int_;
        std::map<std::string, std::vector<std::string>> map_str_vec_str_;
        std::map<std::string, std::map<std::string, std::string>> map_str_map_str_str_;
    };

    {
        std::string json_str(
            "{\"vec_int_ptr\":[1,2,3],\"vec_vec_int\":[[11,22,33],[22,33,44]],\"map_str_vec_str\":{\"k1\":[\"v1\","
            "\"v2\"],\"k2\":["
            "\"v11\",\"v22\"]},\"map_str_map_str_str\":{\"K1\":{\"K11\":\"V11\"},\"K2\":{\"K22\":\"V22\"}}}");
        NestedContainerJsonClass real;
        ASSERT_TRUE(Jsonizable::FromJsonString(json_str, real));
        ASSERT_EQ(std::vector<int>({1, 2, 3}), *real.vec_int_ptr_);
        ASSERT_EQ(std::vector<std::vector<int>>({{11, 22, 33}, {22, 33, 44}}), real.vec_vec_int_);
        ASSERT_EQ((std::map<std::string, std::vector<std::string>>({{"k1", {"v1", "v2"}}, {"k2", {"v11", "v22"}}})),
                  real.map_str_vec_str_);
        ASSERT_EQ((std::map<std::string, std::map<std::string, std::string>>(
                      {{"K1", {{"K11", "V11"}}}, {"K2", {{"K22", "V22"}}}})),
                  real.map_str_map_str_str_);
        ASSERT_EQ(json_str, real.ToJsonString());
    }
}

} // namespace kv_cache_manager