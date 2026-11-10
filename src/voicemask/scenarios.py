"""通話の台本と架空の個人情報を作る。

台本の言い回しの中の {SLOT} を値で埋め、値は1つの部品（Part）として切り出す。
個人情報の部品には、合成で読み間違えないよう、かなの読みを付ける。
氏名は Faker（ja_JP）の姓名一覧から作る。単語登録の評価では、この一覧とは別の出どころの語を使うこと。
"""

from __future__ import annotations

import random
import re

from faker import Faker

from .models import CallScript, Part, Utterance
from .textnorm import digits_to_reading, kata_to_hira

# ---------------------------------------------------------------- 数の読み

_D = ["", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]


def int_reading(n: int) -> str:
    """0〜9999 を読みにする（音便は代表的なものだけ）。"""
    if n == 0:
        return "ぜろ"
    out = ""
    th, n = divmod(n, 1000)
    if th:
        out += {1: "せん", 3: "さんぜん", 8: "はっせん"}.get(th, _D[th] + "せん")
    h, n = divmod(n, 100)
    if h:
        out += {1: "ひゃく", 3: "さんびゃく", 6: "ろっぴゃく", 8: "はっぴゃく"}.get(h, _D[h] + "ひゃく")
    t, n = divmod(n, 10)
    if t:
        out += ("" if t == 1 else _D[t]) + "じゅう"
    return out + _D[n]


_MONTH = {4: "しがつ", 7: "しちがつ", 9: "くがつ"}
_DAY = {
    1: "ついたち", 2: "ふつか", 3: "みっか", 4: "よっか", 5: "いつか", 6: "むいか", 7: "なのか",
    8: "ようか", 9: "ここのか", 10: "とおか", 14: "じゅうよっか", 20: "はつか", 24: "にじゅうよっか",
}


def date_reading(m: int, d: int) -> str:
    return _MONTH.get(m, int_reading(m) + "がつ") + _DAY.get(d, int_reading(d) + "にち")


# ---------------------------------------------------------------- 住所（表記と読みの組）

PLACES = [
    ("東京都", "とうきょうと", [("世田谷区", "せたがやく", ["桜丘", "さくらがおか"]), ("杉並区", "すぎなみく", ["高円寺南", "こうえんじみなみ"]), ("江東区", "こうとうく", ["豊洲", "とよす"]), ("八王子市", "はちおうじし", ["南大沢", "みなみおおさわ"])]),
    ("大阪府", "おおさかふ", [("吹田市", "すいたし", ["江坂町", "えさかちょう"]), ("堺市北区", "さかいしきたく", ["中百舌鳥町", "なかもずちょう"]), ("豊中市", "とよなかし", ["上野西", "うえのにし"])]),
    ("北海道", "ほっかいどう", [("札幌市中央区", "さっぽろしちゅうおうく", ["南三条西", "みなみさんじょうにし"]), ("旭川市", "あさひかわし", ["神楽", "かぐら"])]),
    ("神奈川県", "かながわけん", [("横浜市港北区", "よこはましこうほくく", ["日吉本町", "ひよしほんちょう"]), ("川崎市中原区", "かわさきしなかはらく", ["小杉町", "こすぎまち"]), ("藤沢市", "ふじさわし", ["鵠沼海岸", "くげぬまかいがん"])]),
    ("愛知県", "あいちけん", [("名古屋市千種区", "なごやしちくさく", ["今池", "いまいけ"]), ("岡崎市", "おかざきし", ["康生町", "こうせいちょう"])]),
    ("福岡県", "ふくおかけん", [("福岡市中央区", "ふくおかしちゅうおうく", ["大名", "だいみょう"]), ("北九州市小倉北区", "きたきゅうしゅうしこくらきたく", ["魚町", "うおまち"])]),
    ("宮城県", "みやぎけん", [("仙台市青葉区", "せんだいしあおばく", ["一番町", "いちばんちょう"])]),
    ("広島県", "ひろしまけん", [("広島市中区", "ひろしましなかく", ["紙屋町", "かみやちょう"]), ("福山市", "ふくやまし", ["霞町", "かすみちょう"])]),
    ("京都府", "きょうとふ", [("京都市左京区", "きょうとしさきょうく", ["下鴨", "しもがも"])]),
    ("埼玉県", "さいたまけん", [("さいたま市浦和区", "さいたましうらわく", ["高砂", "たかさご"]), ("川越市", "かわごえし", ["脇田町", "わきたちょう"])]),
    ("千葉県", "ちばけん", [("船橋市", "ふなばしし", ["本町", "ほんちょう"]), ("柏市", "かしわし", ["柏の葉", "かしわのは"])]),
    ("静岡県", "しずおかけん", [("浜松市中央区", "はままつしちゅうおうく", ["鍛冶町", "かじまち"])]),
]

LETTER_READING = dict(zip("abcdefghijklmnopqrstuvwxyz", [
    "えー", "びー", "しー", "でぃー", "いー", "えふ", "じー", "えいち", "あい", "じぇー", "けー", "える", "えむ",
    "えぬ", "おー", "ぴー", "きゅー", "あーる", "えす", "てぃー", "ゆー", "ぶい", "だぶりゅー", "えっくす", "わい", "ぜっと",
]))

EMAIL_DOMAINS = [("example.com", "いぐざんぷる、どっと、こむ"), ("example.jp", "いぐざんぷる、どっと、じぇーぴー"), ("example.net", "いぐざんぷる、どっと、ねっと")]


def spell(s: str) -> str:
    out = []
    for c in s:
        if c.isdigit():
            out.append(_D[int(c)] if c != "0" else "ぜろ")
        elif c == ".":
            out.append("どっと")
        elif c == "_":
            out.append("あんだーばー")
        elif c == "-":
            out.append("はいふん")
        else:
            out.append(LETTER_READING.get(c.lower(), c))
    return "、".join(out)


class Values:
    """1通話分の架空の値。同じ通話では同じ人物の値を使う。"""

    def __init__(self, rng: random.Random):
        self.r = rng
        fk = Faker("ja_JP")
        fk.seed_instance(rng.randrange(1 << 30))
        self.last = fk.last_name_pair()    # (表記, カナ, ローマ字)
        self.first = fk.first_name_pair()
        self.cache: dict[str, Part] = {}

    def get(self, slot: str) -> Part:
        if slot not in self.cache:
            self.cache[slot] = self._make(slot)
        return self.cache[slot]

    def _make(self, slot: str) -> Part:
        r = self.r
        if slot == "NAME":
            return Part(self.last[0] + self.first[0], kata_to_hira(self.last[1] + "、" + self.first[1]), "NAME", slot)
        if slot == "LAST":
            return Part(self.last[0], kata_to_hira(self.last[1]), "NAME", slot)
        if slot == "PHONE":
            n = f"{r.choice(['090', '080', '070'])}-{r.randint(0, 9999):04d}-{r.randint(0, 9999):04d}"
            return Part(n, digits_to_reading(n), "PHONE", slot)
        if slot == "POSTAL":
            n = f"{r.randint(0, 999):03d}-{r.randint(0, 9999):04d}"
            return Part(n, digits_to_reading(n), "POSTAL", slot)
        if slot == "ADDRESS":
            pref, pr, cities = r.choice(PLACES)
            city, cr, (town, tr) = r.choice(cities)
            c, b, g = r.randint(1, 6), r.randint(1, 30), r.randint(1, 20)
            text = f"{pref}{city}{town}{c}丁目{b}番{g}号"
            read = f"{pr}、{cr}、{tr}、{int_reading(c)}ちょうめ、{int_reading(b)}ばん、{int_reading(g)}ごう"
            return Part(text, read, "ADDRESS", slot)
        if slot == "BIRTH":
            y, m, d = r.randint(1950, 2004), r.randint(1, 12), r.randint(1, 28)
            return Part(f"{y}年{m}月{d}日", f"{int_reading(y)}ねん、{date_reading(m, d)}", "BIRTHDATE", slot)
        if slot == "CARD":
            digits = r.choice(["4", "5"]) + "".join(str(r.randint(0, 9)) for _ in range(15))
            g = [digits[i:i + 4] for i in range(0, 16, 4)]
            return Part(" ".join(g), digits_to_reading("-".join(g), "、"), "CARD", slot)
        if slot == "CARD4":
            d = f"{r.randint(0, 9999):04d}"
            return Part(d, digits_to_reading(d), "CARD", slot)
        if slot == "EMAIL":
            local = self.first[2].lower() + r.choice([".", "", "_"]) + self.last[2].lower()
            if r.random() < 0.5:
                local += str(r.randint(1, 99))
            dom, dr = r.choice(EMAIL_DOMAINS)
            return Part(f"{local}@{dom}", f"{spell(local)}、あっとまーく、{dr}", "EMAIL", slot)
        # ---- 個人情報ではない値（消しすぎの確認用）
        if slot == "ORDER":
            n = f"{r.randint(1000, 9999)}-{r.randint(1000, 9999)}"
            return Part(n, digits_to_reading(n), "", slot)
        if slot == "PRICE":
            y = r.choice([980, 1280, 2480, 3980, 5500, 8800])
            return Part(f"{y:,}円", int_reading(y) + "えん", "", slot)
        if slot == "DATE":
            m, d = r.randint(1, 12), r.randint(1, 28)
            return Part(f"{m}月{d}日", date_reading(m, d), "", slot)
        if slot == "ITEM":
            p = r.choice([("電気ケトル", "でんきけとる"), ("加湿器", "かしつき"), ("炊飯器", "すいはんき"), ("折りたたみ傘", "おりたたみがさ"), ("卓上ライト", "たくじょうらいと")])
            return Part(p[0], p[1], "", slot)
        raise KeyError(slot)


# ---------------------------------------------------------------- 台本の型
# 発話は (話者, [言い回しの候補])。O=オペレーター、C=顧客。

O, C = "operator", "customer"

_OPEN = (O, ["お電話ありがとうございます。サポートセンターでございます。", "お待たせいたしました。お客様窓口でございます。"])
_ASK_NAME = (O, ["お客様のお名前をフルネームでお願いいたします。", "ご契約者様のお名前を伺えますでしょうか。"])
_SAY_NAME = (C, ["{NAME}です。", "はい、{NAME}と申します。"])
_CONFIRM = (O, ["{LAST}様ですね。ご本人確認のため、生年月日をお願いいたします。"])
_SAY_BIRTH = (C, ["{BIRTH}です。", "えっと、{BIRTH}です。"])
_ASK_PHONE = (O, ["ご登録のお電話番号をお願いいたします。", "念のため、お電話番号も伺えますか。"])
_SAY_PHONE = (C, ["{PHONE}です。", "携帯で、{PHONE}です。"])
_THANKS = (O, ["確認がとれました。ありがとうございます。"])
_CLOSE = (O, ["お電話ありがとうございました。担当が承りました。", "ほかにご不明な点はございますか。"])
_BYE = (C, ["大丈夫です。ありがとうございました。", "いえ、以上です。"])

SCENARIOS: dict[str, list[tuple[str, list[str]]]] = {
    "address_change": [
        _OPEN,
        (C, ["引っ越したので、住所の変更をお願いしたいんですが。"]),
        _ASK_NAME, _SAY_NAME, _CONFIRM, _SAY_BIRTH, _THANKS,
        (O, ["新しいご住所を、郵便番号からお願いいたします。"]),
        (C, ["郵便番号は{POSTAL}で、{ADDRESS}です。", "{POSTAL}の、{ADDRESS}です。"]),
        (O, ["{DATE}から新しいご住所に切り替わります。"]),
        _CLOSE, _BYE,
    ],
    "card_lost": [
        _OPEN,
        (C, ["クレジットカードをなくしてしまって、止めてほしいんです。"]),
        _ASK_NAME, _SAY_NAME, _CONFIRM, _SAY_BIRTH,
        (O, ["お手元にカード番号の控えはございますか。"]),
        (C, ["はい、カード番号は{CARD}です。", "控えがあります。{CARD}です。"]),
        (O, ["ただいま利用を停止しました。新しいカードは{DATE}ごろにお届けします。"]),
        _ASK_PHONE, _SAY_PHONE, _CLOSE, _BYE,
    ],
    "delivery": [
        _OPEN,
        (C, ["注文した{ITEM}がまだ届かないんですけど。"]),
        (O, ["ご注文番号はお分かりになりますか。"]),
        (C, ["注文番号は{ORDER}です。"]),
        (O, ["お調べします。お届け先のお名前をお願いいたします。"]),
        _SAY_NAME,
        (O, ["{ITEM}、{PRICE}のご注文ですね。お届け先のご住所をお願いいたします。"]),
        (C, ["{ADDRESS}です。"]),
        (O, ["配送が遅れておりまして、{DATE}のお届けになります。"]),
        _CLOSE, _BYE,
    ],
    "member_update": [
        _OPEN,
        (C, ["会員登録のメールアドレスを変えたいんです。"]),
        _ASK_NAME, _SAY_NAME, _CONFIRM, _SAY_BIRTH,
        (O, ["新しいメールアドレスをお願いいたします。"]),
        (C, ["{EMAIL}です。"]),
        _ASK_PHONE, _SAY_PHONE, _THANKS, _CLOSE, _BYE,
    ],
    "reservation": [
        _OPEN,
        (C, ["{DATE}の予約を変更したいんですが。"]),
        (O, ["ご予約のお名前とお電話番号をお願いいたします。"]),
        (C, ["{NAME}で、電話番号は{PHONE}です。"]),
        (O, ["{LAST}様、{DATE}のご予約ですね。ご希望の日にちはございますか。"]),
        (C, ["翌週の同じ曜日でお願いします。"]),
        (O, ["変更いたしました。ご予約金の{PRICE}はそのまま充当いたします。"]),
        _CLOSE, _BYE,
    ],
    "cancel": [
        _OPEN,
        (C, ["サービスを解約したいんです。"]),
        _ASK_NAME, _SAY_NAME, _CONFIRM, _SAY_BIRTH,
        (O, ["お支払いのカードの下4桁をお願いいたします。"]),
        (C, ["下4桁は{CARD4}です。"]),
        (O, ["解約の書類を、ご登録のご住所にお送りします。念のためご住所をお願いいたします。"]),
        (C, ["{ADDRESS}です。"]),
        (O, ["{DATE}付けで解約となります。最後のご請求は{PRICE}です。"]),
        _CLOSE, _BYE,
    ],
}

_SLOT = re.compile(r"\{([A-Z0-9]+)\}")

# 言い回しの読み。漢字を読み違えやすい語だけ読みを足す（合成に渡すときに置き換える）
READING_FIX = {"下4桁": "しもよんけた", "承りました": "うけたまわりました"}


def _fix_reading(s: str) -> str:
    for k, v in READING_FIX.items():
        s = s.replace(k, v)
    return s


def build_call(call_id: str, scenario: str, rng: random.Random) -> CallScript:
    vals = Values(rng)
    utts = []
    for speaker, choices in SCENARIOS[scenario]:
        line = rng.choice(choices)
        parts: list[Part] = []
        pos = 0
        for m in _SLOT.finditer(line):
            if m.start() > pos:
                t = line[pos:m.start()]
                parts.append(Part(t, _fix_reading(t)))
            parts.append(vals.get(m.group(1)))
            pos = m.end()
        if pos < len(line):
            t = line[pos:]
            parts.append(Part(t, _fix_reading(t)))
        utts.append(Utterance(speaker, parts))
    return CallScript(call_id, scenario, utts)


def generate_scripts(n: int, seed: int = 1) -> list[CallScript]:
    rng = random.Random(seed)
    names = list(SCENARIOS)
    return [build_call(f"call{i:04d}", names[i % len(names)], rng) for i in range(n)]
