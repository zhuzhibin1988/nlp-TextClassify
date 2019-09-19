from enum import Enum, unique


@unique
class Dependency(Enum):
    """
      依存句法关系 枚举
    """
    # 主谓关系
    # subject - verb
    # 我送她一束花(我 <– 送)
    SBV = "SBV"

    # 动宾关系
    # 直接宾语，verb - object
    # 我送她一束花(送 – > 花)
    VOB = "VOB"

    # 间宾关系
    # 间接宾语，indirect - object
    # 我送她一束花(送 – > 她)
    IOB = "IOB"

    # 前置宾语
    # 前置宾语，fronting - object
    # 他什么书都读(书 <– 读)
    FOB = "FOB"

    # 兼语
    # double
    # 他请我吃饭(请 – > 我)
    DBL = "DBL"

    # 定中关系
    # attribute
    # 红苹果(红 <– 苹果)
    ATT = "ATT"

    # 状中结构
    # adverbial
    # 非常美丽(非常 <– 美丽)
    ADV = "ADV"

    # 动补结构
    # complement
    # 做完了作业(做 – > 完)
    CMP = "CMP"

    # 并列关系
    # coordinate
    # 大山和大海(大山 – > 大海)
    COO = "COO"

    # 介宾关系
    # preposition - object
    # 在贸易区内(在 – > 内)
    POB = "POB"

    # 左附加关系
    # left adjunct
    # 大山和大海(和 <– 大海)
    LAD = "LAD"

    # 右附加关系
    # right adjunct
    # 孩子们(孩子 – > 们)
    RAD = "RAD"

    # 独立结构
    # independent structure
    # 两个单句在结构上彼此独立
    IS = "IS"

    # 核心关系
    # head
    # 指整个句子的核心
    HED = "HED"