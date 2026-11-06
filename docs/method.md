# 比較の条件と手順

結果そのものは`docs/results/<題材>.md`に置く。ここには、すべての題材に共通する条件を書く。

## 版

| | 版 |
|---|---|
| Ruby | 4.0.7（2026-11-03の計測は4.0.9） / numo-narray-alt 0.11.2（同じく0.11.3） / numo-linalg-alt 0.10.1（同梱OpenBLAS 0.3.34） |
| cumo | master `1c831278`を`vendor/cumo-master/`に入れたもの（版の文字列は0.11.0）。2026-10-04からはrubygemsの0.12.0（`vendor/cumo-0.12.0/`）で、2026-10-18からは0.14.0（`vendor/cumo-0.14.0/`）で、2026-11-03からは0.15.0（`vendor/cumo-0.15.0/`）で、GPUの全条件を取り直した。それまでに、2026-10-03に`aaa3162b`、`1c561d70`、`50e07e88`へ順に入れ替え、関係する経路の一部の条件とメモリの段階表を取り直した（各結果ファイルとREADMEの表に、どの版かを書いてある） |
| Python | 3.14.7（2026-10-14から3.14.8、2026-11-03から3.14.9） / numpy 2.5.3 / scipy 1.18.1 / cupy-cuda13x 14.2.0 |
| CUDA | nvcc 13.4 / CuPyから見たruntime 13.2 / driver 615.71.09（2026-10-11から615.78.08） |
| GPU | RTX 5070 Ti Laptop（sm_120） |

## fixture

- `script/make_fixtures.rb`がNumoのDFloatで作る。乱数は題材ごとに固定したseed（`FixtureGenerators::SEEDS`）から引く。
- 3実装とも同じバイトを読む。RubyとPythonが同じ配列を読めていることは、`test/test_fixtures.rb`で配列ごとのsha256とshapeを比べて確かめている。
- 計測でfp32を使うときは、読んだfp64をbackend側でcastする。fp32のfixtureは別に作らない。

## 関門

- fp64で回す。量は相対残差で書く（`solve`なら`‖A x − b‖ / (‖A‖ ‖x‖ + ‖b‖)`、直交性なら`max |QᵀQ − I|`）。
- tolは、Numoで測った実際の残差の10倍に固定する。測った値は`docs/results/<題材>.md`に残す。
- `script/gate.rb`と`python/gate.py`は、同じ形のJSONを出す（`impl` / `backend` / `dtype` / `size` / `problems`）。題材ごとの`status`は`pass` / `fail` / `skipped`のいずれか。`skipped`は`NotImplementedError`を受けたときで、cumo側の関数がまだ無い場合に当たる。

## 計測

- `script/bench.rb`と`python/bench.py`は、どちらも次の8列のTSVを1行出す。

  ```
  impl  backend  problem  size  dtype  ms/step  best_seconds  all_seconds
  ```

- best-of-3、ウォームアップ1区間。区間内では同期しない。区間の終わりに1回だけ同期する。Cumoは`cudaDeviceSynchronize`を使う。プランでは「結果のスカラーを1個読み戻す」としていたが、ストリームが1本なら待つ範囲は同じで、戻り値の形に依らないのでこちらにした。CuPyは`Stream.null.synchronize()`を使う。
- `bench/run.sh <conditions> <out>`は、Ruby、Python、対照（Rubyをもう一度）の3系列を1ラウンドにまとめる。系列の順序はラウンドごとに回す。ラウンドは11回（0〜10）。`bench/aggregate.py`が先頭を捨て、10ラウンドで集計する。
- 比は「Python ÷ Ruby」の、ラウンドごとの比の中央値。ms/stepの比なので、1を超えればRuby（Cumo / Numo）が速い。後ろに「1を超えたラウンド数 / 10」を付ける。
- 対照はRubyと同じコマンドなので、1をまたぐはず。**またがない条件は、実装以外の何かに偏っているのでREADMEに載せない。**
- GPUの条件とCPUの条件は、別のconditionsファイルに分ける。

### クロックを固定する

```
sudo nvidia-smi -lgc 3090
sudo nvidia-smi -lmc 14001
```

2つは1回ずつ別に実行する。`-lgc 3090`でも3090 MHzは出ず、負荷中は2750〜2805 MHzになる。固定は再起動で外れるので、測る直前に毎回読む。

```
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader   # アイドルで SM 2790〜2805 MHz なら掛かっている
```

GPU条件の`bench/run.sh`は、この読みが2700 MHz未満なら測らずに止まる。終わったら`sudo nvidia-smi -rgc`と`sudo nvidia-smi -rmc`で戻す。計測の前に、GPUを使う他のプロセスが無いことと、load averageが低いことを確かめる。計測中の様子は`$out/monitor.log`に5秒おきに残る。

## 集計の読み方

- 比の列は「中央値、1を超えたラウンド数 / 10、符号検定の両側p」の順に並ぶ。範囲が1をまたぐと「またぐ」と付く。
- 幅の列は系列ごとのmax ÷ min − 1で、5%を超えると「要確認」と付く。その場合は生の系列をラウンド順に見る。メモリの段が一部のプロセスだけ落ちて2群に割れていると、中央値は群の間に落ちる。
