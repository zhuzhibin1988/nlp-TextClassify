from enum import Enum, unique


@unique
class Pos(Enum):
    """
    adjective
    美丽
    """
    a = "a"

    """
    organization name
    保险公司
    """
    ni = "ni"

    """
    other noun - modifier
    大型, 西式
    """
    b = "b"

    """
    location noun
    城郊
    """
    nl = "nl"

    """
    conjunction
    和, 虽然
    """
    c = "c"

    """
    geographical name
    北京
    """
    ns = "ns"

    """
    adverb
    很
    """
    d = "d"

    """
    temporal noun
    近日, 明代
    """
    nt = "nt"

    """
    exclamation
    哎
    """
    e = "e"

    """
    other proper noun
    诺贝尔奖
    """
    nz = "nz"

    """
    morpheme
    茨, 甥
    """
    g = "g"

    """
    onomatopoeia
    哗啦
    """
    o = "o"

    """
    prefix
    阿, 伪
    """
    h = "h"

    """
    preposition
    在, 把
    """
    p = "p"

    """
    idiom
    百花齐放
    """
    i = "i"

    """
    quantity
    个
    """
    q = "q"

    """
    abbreviation
    公检法
    """
    j = "j"

    """
    pronoun
    我们
    """
    r = "r"

    """
    suffix
    界, 率
    """
    k = "k"

    """
    auxiliary
    的, 地
    """
    u = "u"

    """
    number
    一, 第一
    """
    m = "m"

    """
    verb
    跑, 学习
    """
    v = "v"

    """
    general noun
    苹果
    """
    n = "n"

    """
    punctuation    ，。！
    """
    wp = "wp"

    """
    direction noun
    右侧
    """
    nd = "nd"

    """
    foreign words
    CPU
    """
    ws = "ws"

    """
    person name
    杜甫, 汤姆
    """
    nh = "nh"

    """
    non - lexeme
    萄, 翱
    """
    x = "x"

    """
    descriptive words
    瑟瑟，匆匆
    """
    z = "z"