# -*- coding: utf-8 -*-
# @Author  : ZhengHj
# @Time    : 5/17/19 7:17 PM
# @Project : emce
# @File    : log.py
# @IDE     : PyCharm

import logging


def set_logger(console_level=logging.INFO,
               file=True, name="fm", file_level=logging.INFO, mode="w", encoding="utf8"):
    """
    配置日志信息
    :param console_level: logging level，输出到控制台的日志等级，从低到高有NOTSET，DEBUG，INFO，WARNING(WARN)，
    ERROR，CRITICAL(FATAL)，例如：console_level=logging.INFO，这个表示输出INFO及以上等级的日志信息
    :param file: bool，如果file=True，日志将保存到name.log文件，否则不写入到文件，同时`name`，`file_level`，`mode`和`encoding`三个参数失效
    :param name: str，日志名，如果file=True，日志会保存在name.log文件中
    :param file_level: 同console_level，当file=True时，控制写入文件的日志等级
    :param mode: 'w'或'a'，当file=True时，'w'表示写入到文件时会删除原有内容，'a'表示写入文件时在文件最后追加
    :param encoding: 日志文件编码
    :return: Logger
    """
    # 创建Logger对象，设置全局level
    logger = logging.getLogger("speech")
    logger.setLevel(logging.INFO)

    if file:
        # 创建FileHandler，将日志写入到文件中
        handler = logging.FileHandler(name + ".log", mode, encoding)
        handler.setLevel(file_level)
        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        # logger中添加创建FileHandler
        logger.addHandler(handler)

    # 创建StreamHandler，将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)

    # logger中添加StreamHandler
    logger.addHandler(console)

    return logger
