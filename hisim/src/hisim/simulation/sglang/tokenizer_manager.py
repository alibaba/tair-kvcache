from hisim.hook import BaseHook


class C_TokenizerManagerHook(BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"

    @classmethod
    def hook(cls, target):
        original_send_one_request = target._send_one_request

        # When running with blocking mode, send the created time to schedule.
        def wrapped_send_one_request(self, obj, tokenized_obj, created_time):
            if obj.__class__.__name__ == "GenerateReqInput":
                if (
                    tokenized_obj.sampling_params.custom_params is not None
                    and "simulation" in tokenized_obj.sampling_params.custom_params
                ):
                    tokenized_obj.sampling_params.custom_params["simulation"][
                        "server_created_time"
                    ] = created_time
            return original_send_one_request(self, obj, tokenized_obj, created_time)

        target._send_one_request = wrapped_send_one_request
