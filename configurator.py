"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""
"""
这段代码是一个简单的配置器，它用于简化配置的复杂性。它的作用是将配置相关的代码从主程序中分离出来，以便更容易地管理和修改配置。
配置器被称为"configurator.py"。它不是一个Python模块，而是一个独立的脚本文件。
当主程序（"train.py"）需要使用配置时，它会执行配置器中的代码。
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file 参数中不包含等号做配置文件处理
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument 否则做键值对处理
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:] # 移除前缀--
        """
        下面代码，首先检查一个变量key是否存在于全局变量中。
        如果存在，它会尝试对val进行求值，即将其转换为对应的数据类型。
        如果转换成功，它会确保转换后的数据类型与全局变量中key的原本的数据类型相匹配，然后将全局变量key的值覆盖为转换后的值。
        最后，它会打印出覆盖的信息。覆盖原本key的值。如果key不存在于全局变量中，它会抛出一个ValueError异常，表示未知的配置键。

        globals() 是一个内置函数，它返回一个包含全局作用域中所有全局变量和它们的值的字典。
        全局作用域是指在代码中任何地方都可以访问的变量。当你在代码中定义一个变量时，它默认是全局变量，除非你在一个函数内部或其他作用域中定义它。
        """
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val) #对val进行求值，即将其转换为对应的数据类型
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
