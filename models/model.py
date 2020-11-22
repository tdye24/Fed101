from abc import ABC


class Model(ABC):
    def __init__(self):
        pass

    def set_params(self, model_params):
        """
        设置模型参数
        :param model_params: 模型参数
        :return: None
        """
        pass

    def get_params(self):
        """
        获取模型参数
        :return: 模型参数
        """
        pass