# LLM初期開発計画【改良版 v2.0】

## 目的

Kaggle上で初期規模のLLM（1M〜50Mパラメータ）を作成し、推論特化モデルの基礎を確立する。短サイクルでデータ→前処理→学習→評価のループを回し、**再現可能で品質保証された**ワークフローを確立する。

---

## 1. 概要とゴール【改良版】

### 短期ゴール（2〜4週間）

- [ ] Kaggleで動作する前処理ノートブックを完成させる
- [ ] CoTを含む学習データをJSONL形式で1万〜5万サンプル用意する
- [ ] SentencePieceトークナイザーを作成し、トークン化済みデータを保存する
- [ ] **CoT評価ルーブリックを確定**し、品質テンプレートを固定する
- [ ] 初期モデル（5M前後）をKaggleで学習し、バリデーションで安定した改善を確認する

### 中期ゴール（1〜3ヶ月）

- [ ] CoT比率・トークナイザー・学習率の**段階的A/B実験**を複数回実施する
- [ ] 推論品質の**定量評価**（数学正答率、QA EM/F1、CoT妥当性スコア）を確立する
- [ ] 失敗ケースを分類・改善ループへ反映する

### 成功基準【改良版・数値化】

| 基準 | 目標値 |
|------|--------|
| バリデーション損失（PPL） | 初期 PPL から 20% 以上低下、かつ3エポック連続で改善 |
| 数学タスク正答率 | ベースライン比 +10% 以上 |
| QA EM/F1 | EM ≥ 45%, F1 ≥ 60% |
| **CoT妥当性スコア** | **≥ 0.75 / 1.0（後述ルーブリック）** |
| データ品質スコア | 重複率 < 1%, 不適切コンテンツ 0%, エラー率 < 0.5% |

> **注記:** QA EM/F1・数学正答率の目標値は現時点で参照モデル（同規模の既存日本語小型モデル等）による裏付けがない暫定値。Baseline実験（p1_cot20）完走後、実測値をもとに現実的な水準へ再調整すること。

---

## 2. データ設計【改良版】

### 2.1 データ種別と比率（初期）

| データ種別 | 比率 | サンプル数（5万件時） |
|-----------|------|------------|
| 数学・論理問題（CoT付き） | 30% | 15,000 |
| 技術文書・仕様書（要約/質問） | 25% | 12,500 |
| コード＋解説 | 15% | 7,500 |
| 高品質QAペア | 20% | 10,000 |
| 短文会話・対話例 | 10% | 5,000 |

### 2.2 フォーマット

**JSONL（1行1サンプル）**

```json
{
  "id": "math_20250106_0001",
  "input": "3人で1200円を割り勘すると1人いくら？",
  "cot": "合計を人数で割る。1200 ÷ 3 = 400",
  "answer": "400円",
  "meta": {
    "source": "synthetic",
    "lang": "ja",
    "category": "arithmetic",
    "cot_quality_score": 0.85,
    "date_created": "2025-01-06"
  }
}
```

**キーの説明:**
- `id`: `{category}_{YYYYMMDD}_{serial}` 形式で時系列トレーサビリティ確保
- `cot`: 空文字列 `""` ではなく `null` で明示的に「なし」を表現
- `meta.cot_quality_score`: 後述ルーブリックで付与（初期作成時は空でOK）

### 2.3 【新規】CoT評価ルーブリック

CoTの品質を定量化し、データセットの信頼性を確保する。

#### CoT妥当性スコア（0.0〜1.0）

| 観点 | スコア | 判定基準 |
|------|--------|--------|
| **論理正確性** | 0.9〜1.0 | 数学的に完全に正確、演算エラーなし |
| | 0.7〜0.89 | 若干の冗長性や不要ステップあるが、結論は正確 |
| | 0.5〜0.69 | 部分的な誤り or 不完全な説明だが修正可能 |
| | 0.0〜0.49 | 本質的な誤り or 不適切な思考プロセス |
| **完全性** | 0.9〜1.0 | 初期条件から最終答まで全ステップを明示 |
| | 0.7〜0.89 | 1〜2ステップ省略されているが推測可能 |
| | 0.5〜0.69 | 重要ステップが複数省略、理解に努力必要 |
| | 0.0〜0.49 | ほぼ説明なし or 断定のみ |
| **簡潔性** | 0.9〜1.0 | 最小限の言葉で最大限の情報（冗長性なし） |
| | 0.7〜0.89 | 若干の冗長性、でも読みやすい |
| | 0.5〜0.69 | 冗長性が目立つ or 不要なテンプレート |
| | 0.0〜0.49 | 著しく冗長 or 意味不明 |

**最終スコア計算:**
```
cot_quality_score = (論理正確性 + 完全性 + 簡潔性) / 3
```

**採用基準:**
- `cot_quality_score ≥ 0.75`: データセットに採用
- `0.5 ≤ cot_quality_score < 0.75`: 修正リストに追加、Week 2で改良
- `cot_quality_score < 0.5`: 削除または完全な再作成

#### CoT評価スクリプト（Python）【修正版】

**修正点:**
1. `logic`（論理正確性）を**ハードゲート**とする。単純平均だと「論理的に誤っているが完全性・簡潔性が高いCoT」が採用されてしまうため、`logic < 0.5` の場合は他観点の点数に関わらず自動的に `reject` とする。
2. カテゴリ別に固定値を返すのではなく、`qa` / `technical` / `code` など主要カテゴリに応じた実装可能な自動チェック（回答が本文に含まれるか、矛盾する数値がないか等）を用意し、それでも判定できない場合のみ `needs_manual_review` フラグを立てて人手確認に回す（無条件で0.75/0.7を付けない）。
3. `count_reasoning_steps` / `compute_conciseness` / `verify_arithmetic` の実装例を明示する（元案は未定義関数を呼ぶだけだった）。

```python
import re

def verify_arithmetic(cot_text, answer_text):
    """
    CoT内の数式を抽出し、実際に計算して答えと一致するか自動検証する。
    一致すれば1.0、不一致や式が抽出できない場合は0.0。
    """
    expressions = re.findall(r'[\d\.\s\+\-\*/×÷÷]+=\s*[\d\.]+', cot_text)
    if not expressions:
        return 0.0  # 検算できない場合は安全側に倒して0扱い（要手動確認）
    try:
        for expr in expressions:
            lhs, rhs = expr.split('=')
            lhs_norm = lhs.replace('×', '*').replace('÷', '/').strip()
            if abs(eval(lhs_norm) - float(rhs.strip())) > 1e-6:
                return 0.0
        return 1.0 if str(eval_answer_matches(answer_text, expressions)) else 0.5
    except Exception:
        return 0.0

def eval_answer_matches(answer_text, expressions):
    # 最終式の右辺が answer_text に含まれているかの簡易チェック
    last_rhs = expressions[-1].split('=')[-1].strip()
    return last_rhs in answer_text

def count_reasoning_steps(cot_text):
    """
    改行・番号付き箇条書き・接続詞（まず/次に/よって等）で疑似的にステップ数を数える。
    """
    if not cot_text:
        return 0
    line_steps = len([l for l in cot_text.split('\n') if l.strip()])
    connector_steps = len(re.findall(r'(まず|次に|そして|よって|したがって|最後に)', cot_text))
    return max(line_steps, connector_steps, 1 if cot_text.strip() else 0)

def compute_conciseness(cot_text, steps):
    """
    ステップ数あたりの文字数から冗長性を推定する。
    目安: 1ステップあたり 15〜60文字程度を適正範囲とする。
    """
    if not cot_text or steps == 0:
        return 0.0
    chars_per_step = len(cot_text) / steps
    if 15 <= chars_per_step <= 60:
        return 0.95
    elif chars_per_step < 15:
        return 0.6  # 短すぎる（説明不足の疑い）
    elif chars_per_step <= 100:
        return 0.75
    else:
        return 0.4  # 冗長

def evaluate_cot_quality(input_text, cot_text, answer_text, category):
    """
    CoT妥当性スコアを計算する。logic を足切り条件として扱う点が旧版からの変更点。
    """
    scores = {}
    needs_manual_review = False

    # 1. 論理正確性（自動検査。自動判定できないカテゴリは要手動確認フラグを立てる）
    if category == "arithmetic":
        scores['logic'] = verify_arithmetic(cot_text, answer_text)
    elif category in ("qa", "technical", "code"):
        # 自動では正確性を保証できないため、暫定的に中間値を置きつつ必ず手動確認に回す
        scores['logic'] = 0.6
        needs_manual_review = True
    else:
        scores['logic'] = 0.6
        needs_manual_review = True

    # 2. 完全性（ステップ数カウント）
    steps = count_reasoning_steps(cot_text)
    if steps >= 3:
        scores['completeness'] = 0.95
    elif steps == 2:
        scores['completeness'] = 0.75
    elif steps == 1:
        scores['completeness'] = 0.5
    else:
        scores['completeness'] = 0.0

    # 3. 簡潔性（文字数と情報密度）
    scores['conciseness'] = compute_conciseness(cot_text, steps)

    # 論理正確性が閾値未満なら他の観点に関わらずreject（ハードゲート）
    if scores['logic'] < 0.5:
        final_score = scores['logic']
        status = 'reject'
    else:
        final_score = sum(scores.values()) / len(scores)
        status = 'adopt' if final_score >= 0.75 else 'revise'

    return {
        'cot_quality_score': final_score,
        'breakdown': scores,
        'status': status,
        'needs_manual_review': needs_manual_review
    }
```

### 2.4 データ品質ルール【詳細化】

```yaml
重複除去:
  - ハッシュ重複率: < 1%
  - 方法: SHA-256 ハッシュ + n-gram (n=3,4,5) 類似度
  - 閾値: 類似度 > 0.95 は重複と判定

フィルタリング:
  - 政治・宗教・軍事: 除外率 100%
  - 暴力的・性的コンテンツ: 除外率 100%
  - スパム・無意味テキスト: 除外率 100%
  - 著作権懸念（明示的な引用）: 要出所確認

数値検証:
  - 数学問題: 自動検算で正答のみ採用
  - 方程式・公式: ドメイン専門家サンプル抽出で確認

言語品質:
  - エラー率 < 0.5% (文法・タイプミス)
  - 読みやすさスコア > 0.7 (Flesch スコアの日本語版)
```

---

## 3. 前処理パイプライン【改良版】

### 3.1 パイプラインの段階と検査ポイント

```
【段階1: 収集・記録】
    └─ データソースリスト作成
    └─ ライセンス・出所の記録
       [チェックポイント] sources.csv 作成完了

【段階2: クレンジング】
    └─ HTML/MarkDown除去
    └─ 正規化（全角↔半角、改行統一）
    └─ 特殊文字・制御文字除去
       [チェックポイント] エラー率 < 0.5% 確認

【段階3: 構造化】
    └─ 長文をスライディングウィンドウで分割
    └─ 各セグメントに category, source_id, segment_id を付与
       [チェックポイント] 平均トークン長 128±32 確認

【段階4: CoT処理】
    └─ CoT抽出 または 手作成
    └─ CoT評価ルーブリック適用
    └─ cot_quality_score を meta に記入
       [チェックポイント] 採用率 ≥ 80% 確認

【段階5: 翻訳（必要時）】
    └─ 機械翻訳（Google/DeepL）
    └─ サンプル（重要データ：n=100）に対して人手確認
    └─ 翻訳品質スコアを meta に記入
       [チェックポイント] 人手確認サンプルの一致率 ≥ 90%

【段階6: 重複除去】
    └─ SHA-256 ハッシュ一致チェック
    └─ n-gram 類似度 (n=3,4,5) で閾値以上を重複判定
       [チェックポイント] 重複率 < 1% 確認

【段階7: トークナイズ前サンプリング】
    └─ バランス確保: category 別に層化サンプリング
    └─ train:val = 9:1 で分割
       [チェックポイント] train/val サイズと比率を記録

【段階8: トークナイザー学習】
    └─ SentencePiece (unigram) + BPE の並列学習
    └─ 語彙サイズ: 8k, 16k で実験
       [チェックポイント] spm_8k.model, spm_16k.model 保存完了

【段階9: トークン化・保存】
    └─ train.jsonl, val.jsonl をトークン化
    └─ token_ids (list[int]) を追加して保存
       [チェックポイント] v{version}.train.jsonl 作成完了
```

### 3.2 前処理ノートブック実装テンプレート（Kaggle）

```python
# preprocess.ipynb 主要セクション

## セクション1: 設定とログ初期化
import json
import hashlib
from datetime import datetime

DATA_VERSION = "v001"
TIMESTAMP = datetime.now().isoformat()
LOG = {
    'version': DATA_VERSION,
    'timestamp': TIMESTAMP,
    'stages': {}
}

## セクション2: データ収集・記録
sources = {
    'math_synthetic': {'url': '...', 'license': 'CC0'},
    'qa_dataset': {'url': '...', 'license': 'CC-BY-4.0'},
    # ...
}
with open(f'logs/{DATA_VERSION}_sources.json', 'w') as f:
    json.dump(sources, f, indent=2)

## セクション3: クレンジング + チェック
def clean_text(text):
    # HTML除去、正規化処理
    pass

cleaned_data = [clean_text(d) for d in raw_data]
error_count = sum(1 for d in cleaned_data if has_error(d))
LOG['stages']['cleaning'] = {
    'count': len(cleaned_data),
    'error_rate': error_count / len(cleaned_data)
}

## セクション4: CoT評価スコア付与
from eval_cot import evaluate_cot_quality

for item in data:
    if item.get('cot'):
        result = evaluate_cot_quality(
            item['input'], 
            item['cot'], 
            item['answer'],
            item['meta']['category']
        )
        item['meta']['cot_quality_score'] = result['cot_quality_score']
        item['meta']['cot_status'] = result['status']

adopted = [d for d in data if d['meta'].get('cot_status') == 'adopt']
LOG['stages']['cot_evaluation'] = {
    'total': len(data),
    'adopted': len(adopted),
    'adoption_rate': len(adopted) / len(data)
}

## セクション5: 重複除去
seen_hashes = set()
deduplicated = []
for item in data:
    h = hashlib.sha256(item['input'].encode()).hexdigest()
    if h not in seen_hashes:
        seen_hashes.add(h)
        deduplicated.append(item)

dup_rate = 1 - len(deduplicated) / len(data)
LOG['stages']['deduplication'] = {
    'before': len(data),
    'after': len(deduplicated),
    'duplicate_rate': dup_rate
}

## セクション6: トークナイザー学習
import sentencepiece as spm

for vocab_size in [8000, 16000]:
    spm.SentencePieceTrainer.train(
        input='corpus.txt',
        model_prefix=f'spm_{vocab_size}',
        vocab_size=vocab_size,
        model_type='unigram',
        character_coverage=0.995
    )

## セクション7: トークン化 + 保存
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('spm_16000.model')

def tokenize_item(item):
    item['token_ids'] = sp.EncodeAsIds(item['input'])
    item['cot_token_ids'] = sp.EncodeAsIds(item.get('cot', '')) if item.get('cot') else []
    return item

# train/val 分割
train_data = [tokenize_item(d) for d in deduplicated[:int(0.9*len(deduplicated))]]
val_data = [tokenize_item(d) for d in deduplicated[int(0.9*len(deduplicated)):]]

# 保存
def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

save_jsonl(train_data, f'data/{DATA_VERSION}.train.jsonl')
save_jsonl(val_data, f'data/{DATA_VERSION}.val.jsonl')

LOG['stages']['tokenization'] = {
    'tokenizer': 'sentencepiece_unigram_16k',
    'train_samples': len(train_data),
    'val_samples': len(val_data)
}

## セクション8: 最終ログ保存
import json
with open(f'logs/{DATA_VERSION}_preprocessing_log.json', 'w') as f:
    json.dump(LOG, f, indent=2)

print(f"✓ Preprocessing complete: {DATA_VERSION}")
print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
```

---

## 4. モデル設計と学習ループ【改良版】

### 4.1 モデル規模と段階的スケーリング

| フェーズ | パラメータ数 | d_model | n_layers | n_heads | 想定学習時間 |
|---------|-----------|---------|----------|---------|-----------|
| Phase 1 (今) | 5M | 256 | 4 | 4 | 2〜4時間 |
| Phase 2 (Week 2) | 10M | 512 | 6 | 8 | 4〜8時間 |
| Phase 3 (Week 3-4) | 20M | 768 | 8 | 12 | 8〜16時間 |
| Phase 4 (次フェーズ) | 50M | 1024 | 12 | 16 | 16h+ (Kaggle外) |

### 4.2 ハイパーパラメータ【詳細化】

```yaml
基本設定:
  model_type: small_transformer
  d_model: 256          # Phase 1
  n_layers: 4
  n_heads: 4
  d_ff: 1024            # 4 × d_model
  max_seq_length: 512

最適化:
  optimizer: AdamW
  lr: 1.0e-4
  lr_warmup_steps: 1000
  lr_schedule: linear   # warmup 後は linear decay
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8
  weight_decay: 1.0e-5  # L2正則化

バッチ処理:
  batch_tokens: 2048    # トークンベース（可変バッチサイズ）
  grad_accum_steps: 4   # 有効バッチサイズ = 2048 × 4 = 8192 tokens
  max_grad_norm: 1.0    # Gradient clipping

正則化:
  dropout: 0.1          # embedding, attention, FFN に適用
  attention_dropout: 0.1
  residual_dropout: 0.1
  label_smoothing: 0.1  # 交差エントロピー損失

学習:
  epochs: 10
  eval_frequency: 500   # ステップごとに評価
  checkpoint_frequency: 1000
  early_stopping_patience: 5  # 改善なし5評価でストップ
  
Kaggle対策:
  max_session_time_minutes: 540    # 9時間
  checkpoint_save_interval: 1000   # セッション途中でも保存
  resume_from_checkpoint: true     # 中断後の再開機能必須
```

### 4.3 学習ループ（Python擬似コード）

```python
# train_kaggle.ipynb メインループ

import torch
import wandb
from datetime import datetime, timedelta

# 初期化
config = load_config('configs/exp01.yaml')
model = TransformerLM(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'])
scheduler = get_linear_schedule_with_warmup(optimizer, ...)

# wandb 初期化
wandb.init(project='vose-initial-llm', config=config, entity='SEKOIA29')

# チェックポイント復帰チェック
start_step = 0
if config['training'].get('resume_from_checkpoint'):
    ckpt_path = find_latest_checkpoint()
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resumed from {ckpt_path} at step {start_step}")

# セッション時間管理
session_start = datetime.now()
max_session_time = timedelta(minutes=config['kaggle']['max_session_time_minutes'])

# トレーニングループ
global_step = start_step
best_val_loss = float('inf')
no_improve_count = 0

for epoch in range(config['training']['epochs']):
    model.train()
    train_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        # セッション時間チェック
        elapsed = datetime.now() - session_start
        if elapsed > max_session_time * 0.9:  # 90% で警告
            print(f"⚠️ Approaching session limit. Saving checkpoint at step {global_step}")
            save_checkpoint(global_step, model, optimizer, scheduler)
            break
        
        # Forward pass
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids)
        loss = criterion(outputs.logits, labels)
        
        # Backward pass + 勾配蓄積
        loss = loss / config['training']['grad_accum_steps']
        loss.backward()
        train_loss += loss.item()
        
        # 蓄積終了後に最適化ステップ
        if (batch_idx + 1) % config['training']['grad_accum_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['max_grad_norm']
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # 定期的なログ
            if global_step % 100 == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {train_loss / (batch_idx + 1):.4f}")
                wandb.log({
                    'train_loss': train_loss / (batch_idx + 1),
                    'lr': scheduler.get_last_lr()[0],
                    'step': global_step
                })
            
            # 評価
            if global_step % config['training']['eval_frequency'] == 0:
                val_loss, metrics = evaluate(model, val_loader, device)
                print(f"Val Loss (PPL): {val_loss:.4f}, EM: {metrics['em']:.3f}, F1: {metrics['f1']:.3f}")
                
                wandb.log({
                    'val_loss': val_loss,
                    'val_ppl': torch.exp(torch.tensor(val_loss)).item(),
                    'val_em': metrics['em'],
                    'val_f1': metrics['f1'],
                    'step': global_step
                })
                
                # Early stopping + チェックポイント保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    save_checkpoint(global_step, model, optimizer, scheduler, is_best=True)
                    print(f"✓ Best checkpoint saved (val_loss: {val_loss:.4f})")
                else:
                    no_improve_count += 1
                    if no_improve_count >= config['training']['early_stopping_patience']:
                        print(f"Early stopping triggered at step {global_step}")
                        break
            
            # 通常チェックポイント
            if global_step % config['training']['checkpoint_frequency'] == 0:
                save_checkpoint(global_step, model, optimizer, scheduler)
    
    if no_improve_count >= config['training']['early_stopping_patience']:
        break

print(f"✓ Training complete at step {global_step}")
wandb.finish()
```

### 4.4 評価関数【修正版】

**修正点:** 旧版は `argmax(logits)` による teacher-forcing 下の次トークン予測精度を EM/F1/正答率として扱っていたが、これは実際の生成品質を測っていない（1トークン先読み精度と、自己回帰生成した文全体が正解と一致するかは別物）。EM・F1・数学正答率は必ず**モデルに自己回帰生成させた文字列**同士で比較する。損失/PPLのみ teacher-forcing で計算し、それ以外は生成ベースに分離した。

```python
def evaluate(model, val_loader, device, tokenizer, max_gen_tokens=128):
    """
    バリデーション損失（teacher-forcing）と、生成ベースのタスク別メトリクスを計算する。
    """
    model.eval()
    total_loss = 0.0

    generated_texts = []
    reference_texts = []
    categories_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            categories = batch.get('category', ['unknown'] * len(input_ids))

            # 1. 損失・PPL（teacher-forcing）
            outputs = model(input_ids)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            # 2. 生成ベース評価用に、プロンプト部分のみからテキストを自己回帰生成
            prompt_ids = batch['prompt_ids'].to(device)  # 質問/問題文のみ（答え・CoTを含まない）
            gen_ids = model.generate(prompt_ids, max_new_tokens=max_gen_tokens)
            for i in range(len(gen_ids)):
                generated_texts.append(tokenizer.decode(gen_ids[i], skip_special_tokens=True))
                reference_texts.append(batch['answer_text'][i])
                categories_list.append(categories[i])

    avg_loss = total_loss / len(val_loader)
    metrics = {
        'loss': avg_loss,
        'ppl': torch.exp(torch.tensor(avg_loss)).item()
    }

    # QA タスク（EM, F1） — 生成テキストと正解テキストを直接比較
    qa_gen = [g for g, c in zip(generated_texts, categories_list) if c == 'qa']
    qa_ref = [r for r, c in zip(reference_texts, categories_list) if c == 'qa']
    if qa_gen:
        metrics['qa_em'] = compute_em(qa_gen, qa_ref)
        metrics['qa_f1'] = compute_f1(qa_gen, qa_ref)
        metrics['em'] = metrics['qa_em']
        metrics['f1'] = metrics['qa_f1']

    # 数学タスク（正答率） — 生成された最終回答を抽出して正解と比較
    math_gen = [g for g, c in zip(generated_texts, categories_list) if c == 'math']
    math_ref = [r for r, c in zip(reference_texts, categories_list) if c == 'math']
    if math_gen:
        metrics['math_accuracy'] = sum(
            extract_final_answer(g) == r.strip() for g, r in zip(math_gen, math_ref)
        ) / len(math_gen)

    # CoT を含むサンプル数（参考値。品質評価は §2.3 のルーブリックで別途実施）
    cot_count = sum(1 for c in categories_list if c == 'cot')
    if cot_count:
        metrics['cot_count'] = cot_count

    return avg_loss, metrics


def extract_final_answer(generated_text):
    """
    生成テキストの末尾から最終回答部分を抽出する簡易ヘルパー。
    データフォーマットの `answer` フィールドの書式に合わせて調整すること。
    """
    match = re.search(r'([-\d,\.]+円?|[-\d,\.]+)\s*$', generated_text.strip())
    return match.group(1) if match else generated_text.strip()
```

> **注意:** `model.generate()` はビームサーチ/greedy/samplingいずれの方式にするか別途 `configs/exp*.yaml` に明示すること（例: `generation.strategy: greedy`）。評価のたびに生成方式が変わると実験間の比較可能性が失われる。

---

## 5. 実験管理と失敗分析【新規】

### 5.1 実験単位と命名規則

```
exp_{YYYYMMDD}_{phase}_{variant_id}

例:
  exp_20250106_p1_cot20      (Phase 1, CoT比率 20%)
  exp_20250110_p1_cot40      (Phase 1, CoT比率 40%)
  exp_20250113_p2_vocab16k   (Phase 2, 語彙サイズ 16k)
```

### 5.2 実験ログテンプレート（CSV + JSON）

**experiment_log.csv**

```csv
exp_id,date,phase,config_file,data_version,model_params,final_val_loss,val_ppl,qa_em,math_acc,duration_hours,status,notes
exp_20250106_p1_cot20,2025-01-06,1,configs/exp01.yaml,v001,5M,2.456,11.7,0.42,0.68,3.5,completed,baseline experiment
exp_20250110_p1_cot40,2025-01-10,1,configs/exp02.yaml,v001,5M,2.389,10.9,0.45,0.71,3.8,completed,higher CoT ratio
exp_20250113_p1_fail,2025-01-13,1,configs/exp03.yaml,v001,5M,3.102,22.4,0.31,0.55,2.1,failed,OOM after step 500
```

**configs/exp01.yaml**

```yaml
experiment:
  id: exp_20250106_p1_cot20
  date: 2025-01-06
  phase: 1
  objective: "Baseline with 20% CoT ratio"
  
data:
  version: v001
  train_file: data/v001.train.jsonl
  val_file: data/v001.val.jsonl
  cot_ratio: 0.20
  tokenizer: spm_16k
  
model:
  type: small_transformer
  params:
    d_model: 256
    n_layers: 4
    n_heads: 4
    vocab_size: 16000
  
training:
  lr: 1.0e-4
  batch_tokens: 2048
  epochs: 10
  dropout: 0.1
  
expected_results:
  val_loss_baseline: 2.5
  qa_em_target: 0.40
  math_acc_target: 0.65
```

### 5.3 【新規】失敗分析テンプレート

失敗時に自動的に記録し、改善ループへ反映する。

**failure_analysis.json**

```json
{
  "experiment_id": "exp_20250113_p1_fail",
  "timestamp": "2025-01-13T14:23:45Z",
  "error_type": "out_of_memory",
  "error_message": "CUDA out of memory (OOM) at step 512",
  "failure_step": 512,
  "root_cause_hypothesis": "batch_tokens=2048 is too large for 5M model on T4 GPU",
  "affected_component": "training",
  "severity": "critical",
  
  "analysis": {
    "memory_usage_at_failure": "15.8 GB",
    "gpu_memory_limit": "15 GB",
    "config": {
      "batch_tokens": 2048,
      "grad_accum_steps": 4,
      "model_params": "5M"
    }
  },
  
  "immediate_action": "reduce batch_tokens to 1024",
  "next_experiment": "exp_20250114_p1_reduced_batch",
  "expected_improvement": "prevent OOM, maintain training schedule"
}
```

**failure_classification.py**

```python
def classify_failure(error_type, error_message):
    """
    失敗を分類し、改善提案を生成
    """
    categories = {
        'oom': {
            'severity': 'critical',
            'action': 'reduce batch_size or model_params',
            'priority': 1
        },
        'nan_loss': {
            'severity': 'high',
            'action': 'reduce lr, check data for outliers',
            'priority': 2
        },
        'data_error': {
            'severity': 'medium',
            'action': 'review data preprocessing, rerun v{version}',
            'priority': 3
        },
        'timeout': {
            'severity': 'high',
            'action': 'optimize training speed or increase GPU allocation',
            'priority': 2
        }
    }
    
    matched_category = None
    for cat, info in categories.items():
        if cat in error_type.lower():
            matched_category = cat
            break
    
    if matched_category:
        return {
            'category': matched_category,
            **categories[matched_category]
        }
    else:
        return {
            'category': 'unknown',
            'severity': 'medium',
            'action': 'manual investigation required',
            'priority': 4
        }
```

### 5.4 A/B実験デザイン【段階的・詳細版】

#### Week 1: CoT比率の段階的試行

| Exp ID | CoT比率 | 他の設定 | 期待される特性 |
|--------|--------|--------|---------|
| p1_cot00 | 0% | baseline | 下限値 (control) |
| p1_cot20 | 20% | baseline | 推論能力軽微改善 |
| p1_cot40 | 40% | baseline | 推論能力顕著改善 |
| p1_cot60 | 60% | baseline | 過度な冗長性? 確認用 |

**期待される結果:**
- CoT 比率が高いほど、数学タスクでの正答率が上昇
- 一定点（40%前後）を超えると改善が鈍化 or 過学習傾向
- QA タスクでの改善は緩い（推論より記憶タスク）

#### Week 2: トークナイザー + 学習率

| Exp ID | トークナイザー | 語彙 | LR | 期待される特性 |
|--------|---------|------|------|---------|
| p2_vocab08k_lr1e4 | SentencePiece | 8k | 1e-4 | baseline |
| p2_vocab16k_lr1e4 | SentencePiece | 16k | 1e-4 | 표현力向上 |
| p2_vocab16k_lr5e5 | SentencePiece | 16k | 5e-5 | 安定性向上? |
| p2_bpe_16k_lr1e4 | BPE | 16k | 1e-4 | UNK削減、性能改善? |

**期待される結果:**
- 16k語彙 > 8k語彙（表現力）
- BPE vs unigram で UNK率に差、性能差は微細かもしれず

#### Week 3-4: モデルサイズ + データ品質

| Exp ID | Model Size | Data Cleanliness | 期待 |
|--------|-----------|-----------------|------|
| p3_5m_v001 | 5M | 標準 | baseline |
| p3_10m_v001 | 10M | 標準 | 性能向上（過学習傾向?） |
| p3_10m_v002_clean | 10M | 高品質（CoT ≥ 0.8） | 最高性能? |

---

## 6. Kaggle実装チェックリスト【詳細化】

### 段階1: 準備（Day 0-1）

- [ ] Kaggle プロジェクト作成
  - [ ] リポジトリ名: `SEKOIA-vose-initial-llm`
  - [ ] Notebook: `preprocess.ipynb`, `train.ipynb`, `eval.ipynb`
  - [ ] Dataset: `vose-initial-llm-data`

- [ ] CoT評価ルーブリック確定
  - [ ] ルーブリック doc 作成
  - [ ] eval_cot.py 実装（自動採点スクリプト）
  - [ ] 手作成 CoT サンプル 100 件（math x50, qa x50）

- [ ] データソースリスト作成
  - [ ] sources.csv: url, license, source_type
  - [ ] ライセンス確認チェック完了

### 段階2: 前処理（Day 2-5）

- [ ] preprocess.ipynb 完成
  - [ ] セクション1-9 全実装完了
  - [ ] チェックポイント全て green （エラー率 < 0.5% など）
  - [ ] logs/{version}_preprocessing_log.json 生成確認

- [ ] データセット生成
  - [ ] train.jsonl, val.jsonl 生成完了
  - [ ] サンプル行を3件表示して確認

- [ ] SentencePiece学習・保存
  - [ ] spm_8k.model, spm_16k.model 生成完了
  - [ ] test 語彙でエンコード確認 (UNK率 < 3%)

- [ ] データバージョン管理
  - [ ] data/v001.train.jsonl, data/v001.val.jsonl
  - [ ] data/v001_preprocessing_log.json
  - [ ] configs/exp01.yaml

### 段階3: 学習準備（Day 6-7）

- [ ] train_kaggle.ipynb スケルトン完成
  - [ ] モデル定義 (TransformerLM)
  - [ ] DataLoader 実装
  - [ ] optimizer, scheduler 設定

- [ ] wandb 連携設定
  - [ ] Kaggle Secrets に WANDB_API_KEY 登録
  - [ ] wandb.init() 動作確認
  - [ ] 同期テスト（ダミーログ）

- [ ] 1ステップ実行テスト
  - [ ] forward pass 動作確認
  - [ ] loss 計算確認
  - [ ] backward, optimizer.step() 確認

- [ ] チェックポイント保存機構確認
  - [ ] save_checkpoint() 実装
  - [ ] load_checkpoint() 実装
  - [ ] Kaggle Dataset への自動アップロード設定

### 段階4: 第1実験実行（Week 1）

- [ ] Baseline 実験（p1_cot20）実行
  - [ ] 1 エポック完走
  - [ ] checkpoint ファイル生成
  - [ ] wandb グラフ確認

- [ ] 失敗時対応
  - [ ] failure_analysis.json 記録
  - [ ] 次実験の修正内容を config に反映

- [ ] コンテニュアス実験
  - [ ] p1_cot40, p1_cot60 実行
  - [ ] 結果比較用ノートブック作成（結果の可視化）

### 段階5: 評価とログ（Week 2-4）

- [ ] experiment_log.csv 更新（毎実験後）
- [ ] 最良モデルを特定
- [ ] failure_analysis 統計（失敗パターン分類）

---

## 7. ファイル構成【拡張版】

```
/home/claude/SEKOIA-vose-initial-llm/
├── README.md                          # プロジェクト概要
├── EXPERIMENT_LOG.md                  # 実験進捗・結果記録

├── data/
│   ├── v001.train.jsonl              # 学習データ（トークン化済み）
│   ├── v001.val.jsonl                # 検証データ
│   ├── v001_preprocessing_log.json    # 前処理ログ
│   ├── spm_8k.model                  # SentencePiece (8k vocab)
│   ├── spm_16k.model                 # SentencePiece (16k vocab)
│   └── sources.csv                   # データソース・ライセンス記録

├── notebooks/
│   ├── preprocess.ipynb              # データ前処理（Kaggle上で実行）
│   ├── train_kaggle.ipynb            # 学習ループ（Kaggle上で実行）
│   ├── eval.ipynb                    # 評価・可視化
│   └── result_comparison.ipynb        # 実験結果比較

├── src/
│   ├── model.py                      # Transformer LM 定義
│   ├── train.py                      # 学習ループ（汎用版）
│   ├── eval.py                       # 評価関数
│   ├── eval_cot.py                   # CoT評価スクリプト
│   ├── data.py                       # DataLoader 定義
│   └── utils.py                      # ユーティリティ関数

├── configs/
│   ├── exp01.yaml                    # Exp 1: Baseline (CoT 20%)
│   ├── exp02.yaml                    # Exp 2: CoT 40%
│   ├── exp03.yaml                    # Exp 3: CoT 60%
│   ├── exp04.yaml                    # Exp 4: Vocab 16k
│   └── ... (以降の実験)

├── logs/
│   ├── v001_preprocessing_log.json    # 前処理ログ
│   ├── v001_sources.json              # データソース
│   ├── exp01_training.log             # 学習ログ
│   ├── exp01_failure_analysis.json    # 失敗分析（あれば）
│   └── experiment_log.csv             # 全実験サマリー

├── results/
│   ├── checkpoints/
│   │   ├── exp01/
│   │   │   ├── checkpoint_01000.pt
│   │   │   ├── checkpoint_best.pt
│   │   │   └── checkpoint_final.pt
│   │   └── exp02/
│   │       └── ...
│   ├── predictions/
│   │   ├── exp01_val_predictions.json
│   │   └── ...
│   └── plots/
│       ├── exp01_loss_curve.png
│       ├── comparison_cot_ratio.png
│       └── ...

└── docs/
    ├── CoT_RUBRIC.md                 # CoT評価基準（詳細）
    ├── DATA_VERSIONING.md            # データバージョン管理方針
    ├── FAILURE_LOG_TEMPLATE.md       # 失敗分析テンプレート
    └── NEXT_PHASE.md                 # Phase 2 移行ガイド
```

---

## 8. スケジュール（改良版）

| 期間 | Milestone | 主要タスク | 成果物 |
|------|-----------|----------|--------|
| **Day 0-1** | 準備完了 | リポジトリ作成、CoT評価基準確定、100サンプル手作成 | CoT_RUBRIC.md, eval_cot.py, sample_cot.jsonl |
| **Day 2-3** | 前処理 v1 | preprocess.ipynb 実装、SentencePiece学習 | v001.train.jsonl, v001.val.jsonl, spm_*.model |
| **Day 4-5** | 学習準備 | train_kaggle.ipynb スケルトン、wandb連携、1step test | checkpoint ファイル, wandb dashboard |
| **Day 6-7** | Baseline実験 | p1_cot20 実行、logs/結果保存 | exp01_training.log, experiment_log.csv 初版 |
| **Week 2** | CoT比率A/B | p1_cot40, p1_cot60 実行 | experiment_log.csv 更新, comparison plot |
| **Week 3** | 語彙・LR実験 | p2_vocab16k, BPE 試行 | configs/ 拡張, 結果分析ノート |
| **Week 4** | 最終まとめ | 最良モデル選定、次フェーズガイド作成 | NEXT_PHASE.md, final_model.pt |

---

## 9. リスク・コスト管理【詳細化】

### 9.1 Kaggleセッション時間制限【対策強化】

| リスク | 影響度 | 対策 |
|--------|--------|------|
| 学習中にセッション終了 | 高 | checkpointを毎1000ステップで保存 + 再開ロジック実装 |
| 長時間前処理で時間消費 | 中 | 前処理は軽量化、大規模データは段階的に処理 |
| GPU OOM による中断 | 高 | batch_tokens を 2048→1024 にデグレード可能な設定 |

**セッション時間予測:**
```
前処理:     ~1 時間   (data collection, cleaning, tokenization)
1 experiment: ~3-4 時間 (1 epoch ~ 2-3h + evaluation)
→ 9時間内で前処理 + 2 experiments 実行可能
```

### 9.2 データ品質リスク

| リスク | 対策 |
|--------|------|
| CoT ノイズ（誤りが含まれる） | CoT評価スコア (≥0.75) による採用フィルタリング |
| 重複データの過剰 | n-gram + hash によるマルチレイヤー重複除去 |
| 翻訳品質不安 | サンプル抽出 + 人手確認（n=100）、品質スコア記録 |
| バリデーション漏洩 | train/val を9:1 で層化分割、混合なし |

### 9.3 コスト管理

- Kaggle: **無料**（ただしセッション時間制限 9h）
- WandB: **無料プラン** （同期ログ up to 100 runs/month）
- 計算リソース: Kaggle T4 GPU + ローカル CPU で十分

---

## 10. 品質ゲート【新規】

各フェーズ終了時に以下の基準をチェック。達成できなければ改善ループへ。

### Gate 1: Day 7 （データ＆前処理）

- [ ] データセット完成: train 10k+, val 1k+
- [ ] CoT採用率 ≥ 80% （evaluation score ≥0.75）
- [ ] 重複率 < 1%
- [ ] エラー率 < 0.5%
- [ ] preprocessing_log 完全記録

### Gate 2: Week 1 （Baseline学習）

- [ ] Baseline exp (p1_cot20) 完走
- [ ] Val loss（PPL）が初期値から **20% 以上**低下（§1の成功基準と統一。3エポック連続改善も確認）
- [ ] wandb グラフで学習曲線が正常（smooth, decreasing）
- [ ] checkpoint 自動保存の動作確認
- [ ] experiment_log.csv 初回項目の記録完了

### Gate 3: Week 2 （CoT比率実験）

- [ ] 3 experiments (cot20, cot40, cot60) 完了
- [ ] 比較グラフで CoT 比率と性能の相関を確認
- [ ] 最適な CoT 比率を特定（例: 40%）
- [ ] 数学タスク正答率がベースライン比 +5% 以上

### Gate 4: Week 3-4 （モデルスケーリング＆最終化）

- [ ] 2 experiments (vocab 16k, BPE 試行) 完了
- [ ] 最高性能モデルを特定
- [ ] 次フェーズ（50Mモデル、本格学習）へ移行可能な状態
- [ ] NEXT_PHASE.md 作成完了

---

## 11. 次のアクション（優先度順）

### 今すぐ（Day 0）

1. **Kaggle新規Notebook作成**
   - リポジトリ: `SEKOIA-vose-initial-llm`
   - README.md, preprocess.ipynb スケルトン作成

2. **CoT評価基準の確定**
   - CoT_RUBRIC.md ドキュメント作成
   - eval_cot.py の eval_cot_quality() 関数実装

3. **CoTサンプル100件作成**
   - 数学（arithmetic, algebra）50件
   - QA（事実系、推論系）50件
   - 各サンプルに cot_quality_score を計算・付記
   - data/v001_sample_cot.jsonl に保存

### Day 1-3

4. **SentencePiece学習・保存**
   - corpus.txt 作成（全データテキスト）
   - spm_train で 8k, 16k vocab 生成
   - UNK率 < 3% 確認

5. **前処理パイプラインの実装**
   - preprocess.ipynb セクション1-9 全実装
   - logs/ に各チェックポイントの結果を保存

### Day 4-7

6. **学習ループの実装**
   - train_kaggle.ipynb スケルトン → 完全実装
   - wandb 連携、checkpoint 機構、resume ロジック
   - 1 step test で動作確認

7. **第1実験実行（Baseline）**
   - config/exp01.yaml 設定
   - p1_cot20 を実行、logs/ と results/ に保存

---

## 付録A: 失敗事例と対策

### ケース1: NaN Loss（爆発）

**原因:** 学習率が高すぎる、またはデータに外れ値あり

**対策:**
```yaml
# 修正前
lr: 1.0e-4

# 修正後
lr: 5.0e-5
max_grad_norm: 0.5  # より厳しいclipping
data_outlier_check: true
```

### ケース2: OOM（メモリ不足）

**原因:** batch_tokens が大きすぎる

**対策:**
```yaml
# 修正前
batch_tokens: 2048

# 修正後
batch_tokens: 1024
grad_accum_steps: 8  # 有効バッチサイズ同等に
```

### ケース3: Overfitting（過学習）

**原因:** dropout不足、またはバリデーションセット小さすぎ

**対策:**
```yaml
# 修正前
dropout: 0.1
val_ratio: 0.05

# 修正後
dropout: 0.2
val_ratio: 0.1
early_stopping_patience: 3  # より早期に停止
```

---

## 付録B: 参考リンク・資料

- WandB: https://wandb.ai/
- SentencePiece: https://github.com/google/sentencepiece
- Hugging Face Transformers: https://huggingface.co/transformers/
- Kaggle Notebooks: https://www.kaggle.com/code

---

**Version:** 2.0  
**Last Updated:** 2025-01-06  
**Author:** SEKOIA  
**Status:** Ready for Implementation
