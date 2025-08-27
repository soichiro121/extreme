# extreme
ai4.pyを参照して
## 基本問題: build_enhanced_autoencoder 内の「人・道具・要約」3本の流れについて、各層の入出力形を最後まで追跡し、なぜ RepeatVector と TimeDistributed が必要になるのか、言葉と形の両方で説明せよ。さらに、use_attention=True と False の2通りで「結合点の形」がどう変わるかを、破綻しない条件とともに述べよ。  


   
## 応用問題：形が崩れずに学習・判定が最後まで通るための「必要十分条件」をファイル内だけから導け

### 前提

入力は human_input(100×42), bat_input(100×3), features_input(32)。出力は human(100×42), bat(100×3), features(32)。

build_enhanced_autoencoder 内の流れは、human側: CircularConv1D→CircularConv1D→LSTM(64, return_sequences=use_attention)→AttentionMechanism で1本に集約、bat側も同様、features側は全結合で圧縮し、最後に3本を結合して latent を作る。復元は Dense→RepeatVector(100)→LSTM→LSTM→各時点の全結合で human(42) と bat(3) を出す。features は latent から別枝で 32 を出す。

### 要求

「長さ100が壊れない」ための条件を層ごとに列挙し、CircularConv1D の端の処理（左右を継ぎ足してから畳んで、元の長さに切り戻す）と、LSTM の戻り値の形（時間ごと／時間を畳む）の切替が、なぜ AttentionMechanism と両立するかを、戻り値の形の等式で示せ。

RepeatVector で時間長を作り直す設計が「一度だけ必要」な理由を、結合後に時間軸が失われるタイミングと照らして説明せよ。ここで、Attention を使わない分岐でも形が崩れないことを示すこと。

出力3本のうち features は時間を持たないが、human/bat は時間を持つ。この非対称を許すための「結合点の次元条件」と「復元側の分岐条件」を十分条件として定式化せよ。

### 仕上げ

上記を満たさない場合に実際に起こる例外（形不一致、次元超過など）を、どの行で起こりうるかを指摘し、形の等式が破れる最小反例（たとえば return_sequences の真偽の不一致）を1つ構成せよ。
