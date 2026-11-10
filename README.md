# voicemask

通話録音から個人情報（氏名・電話番号・住所など）を見つけ、文字と音声の両方を伏せてから、保存したりLLMに渡したりするための中継処理です。音声認識にはAmiVoice APIを、公式のPythonライブラリ（amivoice-api-client）を通して使います。

- 単語ごとの時刻を使って、文字で見つけた個人情報の位置を音声の区間に戻し、ピー音で消します
- 誤認識で別の言葉に化けた個人情報は、文字からは見つかりません。そこで、「お名前を」などと尋ねた直後でconfidenceの低い単語も伏せます
- 架空の個人情報を入れた通話を合成音声で作り、正解の時刻と比べて漏れ率と消しすぎ率を測ります

法令への対応を保証するものではありません。伏せ損ねは0になりません。

見本（伏せる前と後の音声、伏せた文字）：[docs/index.html](https://subara3.github.io/voicemask/)

## 結果（評価用200通話、2026年11月）

台本300本のうち、規則を直すために中身を見たのは0〜99番だけです。下の数字は、中身を見ていない100〜299番の200本（個人情報833件）で測りました。音声はGoogle Cloud TTS（Chirp 3 HD）で合成し、AmiVoiceの会話_汎用（ハイブリッド、ログ保存なし）で認識しています。

| 条件 | 区間漏れ率 | 消しすぎ率 | Claudeに届く個人情報 |
|---|---|---|---|
| B：規則とGiNZA | 5.3% | 2.5% | 0件 |
| C2：Bに、尋ねた答えの文を足す | 2.2% | 3.4% | 0件 |
| C2のあと、伏せた文字だけをClaude Haikuに見せて取りこぼしを足す | 0.7% | 3.5% | 15件 |
| （比較）伏せる前の文字をClaude Haikuに見せる | 1.0% | 1.8% | 833件 |

- 電話音質（8kHz mu-law）にしても漏れは増えませんでした（同じ150本でC2の区間漏れ率 2.9% → 1.9%）
- 顧客台帳の氏名を通話ごとに単語登録すると（条件D）、氏名が台本どおりの表記で認識された件数は184件中125件から172件に増えました。伏せ漏れはもともと少なく、C2の区間漏れ率は2.2%のままでした（同じ100本）
- E2E（日本語E2E_汎用）は単語の時刻が約0.45秒遅れて出ます。時刻を450ms前にずらしても、C2の区間漏れ率は5.7%でした（ハイブリッドは同じ100本で2.2%）

## 準備

Python 3.11以上が要ります（公式ライブラリの条件）。

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS・Linux は source .venv/bin/activate
pip install -e ".[ner,llm,plot,dev]"
copy .env.example .env            # APIキーを入れる
pytest -q
```

`.env` は、コマンドを実行したフォルダから読み込みます。

## オフラインで一巡させる

APIキーがなくても、疑似の合成音声と疑似の認識結果で全部の工程を通せます。出てくる数値は本物ではありません。

```bash
voicemask gen-scripts --n 12 --out data/dry/scripts.jsonl
voicemask synth --scripts data/dry/scripts.jsonl --out data/dry --tts fake --variants clean,tel
voicemask fake-recognize --truth data/dry/truth.jsonl --out runs/dry
voicemask eval --truth data/dry/truth.jsonl --rec runs/dry --out results/dry --sweep --plot
voicemask stream-eval --truth data/dry/truth.jsonl --rec runs/dry --out results/dry_stream --simulate
```

## 本物のAPIで6通話だけ確かめる

```bash
voicemask gen-scripts --n 6 --out data/w1/scripts.jsonl
voicemask synth --scripts data/w1/scripts.jsonl --out data/w1 --tts google --variants clean,tel
voicemask recognize --truth data/w1/truth.jsonl --engine hybrid --out runs/w1_hybrid
voicemask recognize --truth data/w1/truth.jsonl --engine e2e --max-chunk-s 18 --out runs/w1_e2e
voicemask timing --truth data/w1/truth.jsonl --rec runs/w1_hybrid
voicemask timing --truth data/w1/truth.jsonl --rec runs/w1_e2e
voicemask recognize --truth data/w1/truth.jsonl --engine hybrid --variant tel --out runs/w1_hybrid
voicemask stream --truth data/w1/truth.jsonl --engine hybrid --out runs/w1_ws --limit 1
```

| 確かめること | 見る場所 | 続けてよい目安 |
|---|---|---|
| 単語ごとの時刻とconfidenceが返るか | `runs/*/call0000__clean.json` の `tokens` | 両エンジンで値が入っている |
| 時刻のずれ | `voicemask timing` の `abs_p90` | 200ms以下。超えたら `--pad-ms` を広げる |
| 電話音質で認識できるか | `--variant tel` の結果 | 8kHzのmu-lawで結果が返る |
| 合成の読み | `data/w1/audio/*.wav` を聞く | 氏名・番号を台本どおりに読んでいる |
| WebSocket | `runs/w1_ws/*.json` の `events` | `final` が残っている |

認識済みのファイルはもう一度送りません。合成結果は `data/*/tts_cache` に残るので、同じ文を合成し直しても課金されません。

## 本番の流れ

```bash
voicemask gen-scripts --n 300 --out data/v1/scripts.jsonl
voicemask synth --scripts data/v1/scripts.jsonl --out data/v1 --tts google --variants clean,noise20,noise10,tel

voicemask recognize --truth data/v1/truth.jsonl --engine hybrid --out runs/hybrid
voicemask recognize --truth data/v1/truth.jsonl --engine e2e --max-chunk-s 18 --out runs/e2e --start 100 --limit 100

voicemask eval --truth data/v1/truth.jsonl --rec runs/hybrid --out results/hybrid_clean --sweep --plot
voicemask eval --truth data/v1/truth.jsonl --rec runs/e2e --out results/e2e_clean --sweep --plot

voicemask mask --truth data/v1/truth.jsonl --rec runs/hybrid --out masked/hybrid --condition C
voicemask recheck --truth data/v1/truth.jsonl --masked masked/hybrid --out results/recheck_hybrid.json

voicemask stream --truth data/v1/truth.jsonl --engine hybrid --out runs/ws_hybrid --limit 30
voicemask stream-eval --truth data/v1/truth.jsonl --rec runs/ws_hybrid --out results/stream_hybrid
```

### 比べる条件

| 条件 | 中身 |
|---|---|
| A | 文字だけ伏せる（音声は残る） |
| B | 規則とGiNZAで見つけた単語の時刻で、音声も伏せる |
| C | Bに、尋ねた直後でconfidenceの低い単語を足す。番号の一部が低ければ、番号の並び全体を伏せる |
| C2 | Bに、尋ねた答えの文を丸ごと足す |
| D | C2を、顧客台帳の氏名を単語登録して認識し直した結果で回す |

### 条件D（単語登録）

コールセンターでは、着信番号から顧客台帳を引けることが多いので、その顧客の氏名を通話ごとにAmiVoiceへ単語登録してから認識します。ここでは台本を顧客台帳の代わりにしています（`--crm-scripts`）。

```bash
voicemask recognize --truth data/v1/truth.jsonl --engine hybrid --crm-scripts data/v1/scripts.jsonl --out runs/hybrid_D --start 100 --limit 100
```

決まった語の一覧を登録するときは、`表記,読み` のCSVから作れます。

```bash
voicemask profile-words --csv words.csv --out words.txt              # ハイブリッド用（表記 読み）
voicemask profile-words --csv words.csv --out words_e2e.txt --e2e    # E2E用（表記  強調度。強調度は0〜1）
voicemask recognize --truth data/v1/truth.jsonl --engine hybrid --profile-words words.txt --out runs/hybrid_pw
```

### 録音1本を伏せる

```bash
voicemask run 録音.wav --out masked/demo --llm-after --summarize --delete-source
```

伏せた音声（`*.masked.wav`）、伏せた文字（`*.masked.txt`）、監査ログ（`*.audit.json`）を書き出します。`--llm-after` を付けると、伏せたあとの文字だけをClaudeに見せ、手元で取りこぼした個人情報を足します。`--summarize` を付けると、伏せた文字だけをClaudeに渡して対応記録を作ります。`--delete-source` を付けると、伏せたあとに元の音声を消します。

## AmiVoiceの指定で決めていること

| 項目 | 指定 | 理由 |
|---|---|---|
| `confidenceLevel` | 0 | 既定の0.1では、信頼度の低い発話が結果ごと捨てられる。文字に出ないまま、音声に個人情報が残る |
| ログ | 保存なしの接続先（`logging_opt_out=True`） | 認識を頼む先にも残さない |
| E2Eの長い音声 | `--max-chunk-s 18` で無音の位置で区切って送る | 20秒を超えると、結果の一部が繰り返されることがある |
| E2Eの時刻 | 伏せる区間を450ms前にずらす（自動） | E2Eは単語の時刻が約0.45秒遅れて出る |
| 待ち時間 | 接続15秒、応答3分で打ち切り、2回まで送り直す | 応答が返らないまま待ち続けたことがある |
| 電話音質 | 8kHzのmu-lawのWAV。WebSocketは `MULAW` | 実際の電話の音声に近づける |

## 検出器

| 名前 | 中身 | 外部に送るもの |
|---|---|---|
| rules | 番号・生年月日・メールアドレス・住所の形、「様」の前の名前 | なし |
| ner | GiNZAの固有表現（人名・地名） | なし |
| context | 尋ねた直後でconfidenceの低い単語、検出した範囲の隣でconfidenceの低い単語 | なし |
| llm | Claudeに単語の番号で範囲を答えさせる（`eval --llm`） | 伏せる前の文字 |
| llm（伏せたあと） | 手元で伏せた語を「［伏］」に置き換えてからClaudeに見せる（`run --llm-after`） | 手元で取りこぼした分だけ |

伏せる前の文字をそのまま送る llm は、「LLMに渡す前に伏せる」という目的とは逆向きなので、比較のためだけに使います。

## 指標

| 指標 | 意味 |
|---|---|
| 漏れ率（時間） | 正解の個人情報の音声のうち、伏せ損ねた長さの割合 |
| 区間漏れ率 | 個人情報1件ごとに、90%未満しか伏せられなかった件数の割合 |
| 文字漏れ率 | 正解の区間に半分以上かかる内容語が、伏せずに残った件数の割合 |
| 消しすぎ率 | 個人情報でない発話のうち、伏せた長さの割合 |
| 適合率 | 伏せた長さのうち、本当に個人情報だった割合 |

## 作りについて

- 正解の時刻：発話を「個人情報の前・個人情報・後」の部品に分けて別々に合成し、つないだ位置を正解にします。合成音声は部品の頭に息の音が付くので、採点のときは雑音のない版の音声から声のある範囲（ピークから-30dB）に詰めます。代わりに、部品の継ぎ目の抑揚が少し不自然になります
- 姓と名のあいだの間：伏せた区間どうしのあいだに、伏せていない声の単語がなければ1つにつなぎます
- 同期HTTPの区切り：同期HTTPは1ファイル全体を1区間で返すので、単語のあいだの無音（600ms以上）で発話に分け直します
- 数字の揺れ：「ぜろきゅうぜろ」「〇九〇」「０９０」は、照合の前に算用数字にそろえます。「まる」1語だけのときは数字とみなしません
- 監査ログ：電話番号のように候補の少ない値は、鍵なしのハッシュだと総当たりで戻せるので、HMACにしています
- リアルタイム：音声をDミリ秒遅らせて流すと、確定結果が届いた時刻aより後に外へ出る部分だけを伏せられます。ためる時間ごとに漏れ率を出します

## 公開について

`docs/` の見本の音声6本は、Google Cloud TTS で合成した音声です。人名・住所・番号はすべて架空です。それ以外の合成音声は公開していません。台本と生成スクリプトから、手元で作り直せます。

`data/` `runs/` `masked/` `results/` `.env` は `.gitignore` に入っています。

## ライセンス

MIT License（[LICENSE](LICENSE)）
