from abc import ABC, abstractmethod

class JsonData(ABC):
    @abstractmethod
    def to_json_data(self) -> dict:
        pass

    @abstractmethod
    def check(self) -> bool:
        pass

    @staticmethod
    def expect_exist(key: str, json_data: dict, type_):
        if not JsonData.maybe_exist(key, json_data, type_):
            raise RuntimeError(f"not exist key : [{key}] in json")
        return True

    @staticmethod
    def maybe_exist(key: str, json_data: dict, type_):
        if key in json_data:
            if not isinstance(json_data[key], type_):
                raise RuntimeError(f"field [{key}] expect type : [{type_}], real [{type(json_data[key])}]")
            return True
        return False
    
    def __repr__(self) -> str:
        return str(self.to_json_data())