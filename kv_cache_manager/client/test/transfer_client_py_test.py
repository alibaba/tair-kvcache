#!/usr/bin/env python3

"""
Python版本的TransferClient测试
参考C++版本的transfer_client_test.cc
使用unittest框架
"""

import sys
import os
import tempfile
import ctypes
import unittest
from kv_cache_manager.client.pybind import kvcm_py_client

class TransferClientPyTest(unittest.TestCase):
    """TransferClient Python测试类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {self.temp_dir}")
        
        # 准备测试数据
        self.test_data1 = b"test key1"
        self.test_data2 = b"test key2"
        
        # 创建测试文件
        self.test_file1 = os.path.join(self.temp_dir, "key1")
        self.test_file2 = os.path.join(self.temp_dir, "key2")
        
        with open(self.test_file1, 'wb') as f:
            f.write(self.test_data1)
        
        with open(self.test_file2, 'wb') as f:
            f.write(self.test_data2)
        
        # 构建客户端配置
        self.client_config = f'''{{
            "instance_group": "test_group",
            "instance_id": "test_instance",
            "block_size": 16,
            "sdk_config": {{
                "thread_num": 4,
                "queue_size": 1000,
                "sdk_config": [],
                "timeout_config": {{
                    "get_timeout_ms": 5000,
                    "put_timeout_ms": 10000
                }}
            }},
            "location_spec_infos": {{
                "tp0": 1024
            }}
        }}'''
        
        # 创建InitParams
        self.init_params = self._create_init_params("tp0")
        
        # 准备locations用于后续测试
        self.uri_str_vec = [
            f"file://test_nfs/{self.test_file1}?blkid=0&size=1024",
            f"file://test_nfs/{self.test_file2}?blkid=0&size=1024"
        ]

    def _create_init_params(self, spec_name):
        """创建初始化参数"""
        
        init_params = kvcm_py_client.InitParams()
        init_params.role_type = kvcm_py_client.RoleType.WORKER
        
        # 分配内存
        buffer_size = 1024 * 1024
        buffer = ctypes.create_string_buffer(buffer_size)
        
        # 获取缓冲区的地址
        buffer_addr = ctypes.addressof(buffer)
        
        regist_span = kvcm_py_client.RegistSpan()
        regist_span.base = buffer_addr  # 使用地址值
        regist_span.size = buffer_size
        init_params.regist_span = regist_span
        
        init_params.self_location_spec_name = spec_name
        init_params.storage_configs = f'''[
            {{
                "type": "file",
                "global_unique_name": "test_nfs",
                "storage_spec": {{
                    "root_path": "{self.temp_dir}/",
                    "key_count_per_file": 5
                }}
            }}
        ]'''
        
        return init_params

    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_create_client_success(self):
        """测试1: 正常创建客户端"""
        
        client = kvcm_py_client.TransferClient.Create(self.client_config, self.init_params)
        self.assertIsNotNone(client, "TransferClient should be created successfully")

    def test_create_client_with_invalid_config(self):
        """测试2: 使用无效配置创建客户端"""
        
        invalid_config = "{}"
        client = kvcm_py_client.TransferClient.Create(invalid_config, self.init_params)
        self.assertIsNone(client, "TransferClient should return None for invalid config")

    def test_create_client_with_empty_location_spec_name(self):
        """测试3: 使用空的location spec name创建客户端"""
        
        init_params_empty_spec = kvcm_py_client.InitParams()
        init_params_empty_spec.role_type = kvcm_py_client.RoleType.WORKER
        
        # 分配内存
        buffer_size = 1024 * 1024
        buffer = ctypes.create_string_buffer(buffer_size)
        buffer_addr = ctypes.addressof(buffer)
        
        regist_span = kvcm_py_client.RegistSpan()
        regist_span.base = buffer_addr
        regist_span.size = buffer_size
        init_params_empty_spec.regist_span = regist_span
        init_params_empty_spec.self_location_spec_name = ""
        init_params_empty_spec.storage_configs = self.init_params.storage_configs
        
        client = kvcm_py_client.TransferClient.Create(self.client_config, init_params_empty_spec)
        self.assertIsNone(client, "TransferClient should return None for empty spec name")

    def test_create_client_with_wrong_role_type(self):
        """测试4: 使用不支持的角色类型创建客户端"""
        
        init_params_wrong_role = kvcm_py_client.InitParams()
        init_params_wrong_role.role_type = kvcm_py_client.RoleType.SCHEDULER
        
        # 分配内存
        buffer_size = 1024 * 1024
        buffer = ctypes.create_string_buffer(buffer_size)
        buffer_addr = ctypes.addressof(buffer)
        
        regist_span = kvcm_py_client.RegistSpan()
        regist_span.base = buffer_addr
        regist_span.size = buffer_size
        init_params_wrong_role.regist_span = regist_span
        init_params_wrong_role.self_location_spec_name = "tp0"
        init_params_wrong_role.storage_configs = self.init_params.storage_configs
        
        client = kvcm_py_client.TransferClient.Create(self.client_config, init_params_wrong_role)
        self.assertIsNone(client, "TransferClient should return None for wrong role type")

    def test_create_with_empty_address(self):
        """测试5: 使用空地址创建客户端"""
        
        
        # 创建不包含地址信息的配置
        client_config = '''{
            "instance_group": "group",
            "instance_id": "instance",
            "block_size": 128,
            "sdk_config": {},
            "model_deployment": {
                "model_name": "test_model",
                "dtype": "FP8",
                "use_mla": false,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1
            },
            "location_spec_infos": {
                "tp0": 1024
            }
        }'''
        
        client = kvcm_py_client.TransferClient.Create(client_config, self.init_params)
        self.assertIsNotNone(client, "TransferClient should be created successfully with empty address")

    def test_load_kv_caches(self):
        """测试6: 加载KV缓存"""
        
        
        client = kvcm_py_client.TransferClient.Create(self.client_config, self.init_params)
        self.assertIsNotNone(client, "TransferClient should be created successfully")
        
        # 创建BlockBuffers
        block_buffers = [kvcm_py_client.BlockBuffer(), kvcm_py_client.BlockBuffer()]
        
        result = client.LoadKvCaches(self.uri_str_vec, block_buffers)
        self.assertEqual(result, kvcm_py_client.ClientErrorCode.ER_OK, 
                         "LoadKvCaches should return ER_OK")

    def test_save_kv_caches(self):
        """测试7: 保存KV缓存"""
        
        
        client = kvcm_py_client.TransferClient.Create(self.client_config, self.init_params)
        self.assertIsNotNone(client, "TransferClient should be created successfully")
        
        # 创建BlockBuffers
        block_buffers = [kvcm_py_client.BlockBuffer(), kvcm_py_client.BlockBuffer()]
        
        result = client.SaveKvCaches(self.uri_str_vec, block_buffers)
        
        self.assertEqual(result[0], kvcm_py_client.ClientErrorCode.ER_OK, 
                         "SaveKvCaches should return ER_OK")
        self.assertEqual(len(result[1]), len(self.uri_str_vec), 
                         "SaveKvCaches should return uri_str_vec size results")

    def test_empty_locations(self):
        """测试8: 空位置列表测试"""
        
        
        client = kvcm_py_client.TransferClient.Create(self.client_config, self.init_params)
        self.assertIsNotNone(client, "TransferClient should be created successfully")
        
        # 空的位置列表和缓冲区列表
        empty_locations = []
        empty_block_buffers = []
        
        # 测试LoadKvCaches
        result_load = client.LoadKvCaches(empty_locations, empty_block_buffers)
        self.assertEqual(result_load, kvcm_py_client.ClientErrorCode.ER_INVALID_PARAMS, 
                         "LoadKvCaches should return ER_INVALID_PARAMS for empty uri_str_vec")
        
        # 测试SaveKvCaches
        result_save = client.SaveKvCaches(empty_locations, empty_block_buffers)
        self.assertEqual(result_save[0], kvcm_py_client.ClientErrorCode.ER_INVALID_PARAMS, 
                         "SaveKvCaches should return ER_INVALID_PARAMS for empty uri_str_vec")
        self.assertTrue(len(result_save[1]) == 0, 
                        "SaveKvCaches should return empty results for empty uri_str_vec")

    def test_many_locations(self):
        """测试9: 大量位置列表测试"""
        
        
        client = kvcm_py_client.TransferClient.Create(self.client_config, self.init_params)
        self.assertIsNotNone(client, "TransferClient should be created successfully")
        
        # 创建大量位置和缓冲区
        uri_str_vec = []
        block_buffers = []
        
        for i in range(100):
            uri_str = f"file://test_nfs/{self.temp_dir}/key_{i}?blkid=0&size=1024"
            # 创建临时文件
            temp_file = os.path.join(self.temp_dir, f"key_{i}")
            with open(temp_file, 'wb') as f:
                f.write(f"test data for key_{i}".encode())
            
            uri_str_vec.append(uri_str)
            block_buffers.append(kvcm_py_client.BlockBuffer())
        
        # 测试SaveKvCaches
        result = client.SaveKvCaches(uri_str_vec, block_buffers)
        self.assertEqual(result[0], kvcm_py_client.ClientErrorCode.ER_OK, 
                         "SaveKvCaches should return ER_OK for many uri_str_vec")
        self.assertEqual(len(result[1]), len(uri_str_vec), 
                         "SaveKvCaches should return correct number of results")

    def test_mismatched_locations_and_buffers(self):
        """测试10: 位置和缓冲区数量不匹配测试"""
        
        
        client = kvcm_py_client.TransferClient.Create(self.client_config, self.init_params)
        self.assertIsNotNone(client, "TransferClient should be created successfully")
        
        # 创建只有一个缓冲区，但有两个位置的情况
        block_buffers = [kvcm_py_client.BlockBuffer()]  # 只有一个缓冲区
        
        result = client.LoadKvCaches(self.uri_str_vec, block_buffers)
        self.assertEqual(result, kvcm_py_client.ClientErrorCode.ER_INVALID_PARAMS, 
                         "LoadKvCaches should return ER_INVALID_PARAMS for mismatched uri_str_vec and buffers")

    def test_block_buffer_usage(self):
        """测试11: BlockBuffer使用测试"""
        
        
        client = kvcm_py_client.TransferClient.Create(self.client_config, self.init_params)
        self.assertIsNotNone(client, "TransferClient should be created successfully")
        
        # 获取测试数据长度
        len1 = len(self.test_data1)
        len2 = len(self.test_data2)
        
        # 创建内存缓冲区
        total_size = 1024 * 1024
        get_buffer = (ctypes.c_char * total_size)()
        
        # 复制测试数据到缓冲区
        ctypes.memmove(get_buffer, self.test_data1, len1)
        ctypes.memmove(ctypes.addressof(get_buffer) + len1, self.test_data2, len2)
        
        # 创建BlockBuffer
        buffer1 = kvcm_py_client.BlockBuffer()
        buffer2 = kvcm_py_client.BlockBuffer()
        
        # 配置buffer1
        iov1 = kvcm_py_client.Iov()
        iov1.type = kvcm_py_client.MemoryType.CPU
        iov1.base = ctypes.addressof(get_buffer)
        iov1.size = len1
        iov1.ignore = False
        buffer1.iovs = [iov1]  # 直接赋值列表
        
        # 配置buffer2
        iov2 = kvcm_py_client.Iov()
        iov2.type = kvcm_py_client.MemoryType.CPU
        iov2.base = ctypes.addressof(get_buffer) + len1
        iov2.size = len2
        iov2.ignore = False
        buffer2.iovs = [iov2]  # 直接赋值列表
        
        block_buffers = [buffer1, buffer2]
        
        # 加载KV缓存
        result = client.LoadKvCaches(self.uri_str_vec, block_buffers)
        self.assertEqual(result, kvcm_py_client.ClientErrorCode.ER_OK, 
                         "LoadKvCaches should return ER_OK")
        
        # 验证数据是否正确加载
        # 从缓冲区中读取数据并验证
        loaded_data1 = ctypes.string_at(get_buffer, len1)
        loaded_data2 = ctypes.string_at(ctypes.addressof(get_buffer) + len1, len2)
        
        self.assertEqual(loaded_data1, self.test_data1, 
                         f"Loaded data1 {loaded_data1} does not match original {self.test_data1}")
        self.assertEqual(loaded_data2, self.test_data2, 
                         f"Loaded data2 {loaded_data2} does not match original {self.test_data2}")
        
        # 额外验证：检查缓冲区中的数据是否与预期一致
        # 通过切片方式验证
        buffer_bytes = bytearray(get_buffer)
        original_combined = self.test_data1 + self.test_data2
        self.assertEqual(buffer_bytes[:len(original_combined)], original_combined,
                         "Combined buffer content does not match original data")


if __name__ == '__main__':
    # 运行测试
    unittest.main()
