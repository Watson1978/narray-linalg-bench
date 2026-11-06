# メモリ使用量とリークの判定

2026-10-02に調べた。cumoはmaster `1c831278`（vendor）、CuPyは14.2.0、numo-linalg-alt 0.10.1、numpy 2.5.3。

## 結論

- **メモリリークは見つからなかった。** 7題材・12条件で、Cumo、CuPy、Numo、numpyの4実装とも、ウォームアップ後に8N回（Nは最初の2.5秒で回せた回数）回しても、RSSとGPUのメモリプールの使用量は増え続けなかった。
- **Cumoのプロセスが0.9〜1.3 GBと大きいのは、起動時の固定の上乗せである。** `require 'cumo/narray'`だけで430 MB増える（CuPyの`import cupy`は200 MB）。`cumo.so`がCUDAのライブラリ（cuSOLVER、cuDNN、cuBLAS、cuBLASLt、NVRTC、nvJitLink、cuSPARSE）を直接リンクしていて、読み込んだ時点でそれらが載るためである。

## リークの判定

手順は次のとおり。スクリプトはscratchpadの`mem/leak.rb`と`mem/leak.py`で、コミットはしていない。

1. 計測と同じ`bench_step`を20回回してウォームアップする。
2. 2.5秒で回せた回数をNとし、0、N、2N、4N、8N回の時点で、RSS（`/proc/self/status`のVmRSS）と、GPUのメモリプールの使用中バイト数（Cumoは`Cumo::CUDA::MemoryPool.used_bytes`、CuPyは`get_default_memory_pool().used_bytes()`）を記録する。
3. 各記録の直前に`GC.start`（Pythonは`gc.collect()`）を呼ぶ。ここで増え続けていなかったので、GCを挟まない系列は取っていない。

fp32、GPUは`GPU=1`。

| 題材 | variant | size | 実装 | 8N | RSS MB（0 / N / 2N / 4N / 8N） | プール使用中MB（同） |
|---|---|---|---|---|---|---|
| gaussian_process | default | n1024 | cumo | 31536 | 949 / 949 / 949 / 949 / 949 | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 |
| gaussian_process | default | n1024 | cupy | 23688 | 768 / 768 / 768 / 768 / 768 | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 |
| gaussian_process | default | n1024 | numo | 904 | 32 / 32 / 32 / 32 / 32 | — |
| gaussian_process | default | n1024 | numpy | 1256 | 119 / 119 / 119 / 119 / 119 | — |
| vandermonde_lstsq | lstsq | m16384_n64 | cumo | 12976 | 976 / 976 / 976 / 976 / 976 | 8.1 / 8.1 / 8.1 / 8.1 / 8.1 |
| vandermonde_lstsq | lstsq | m16384_n64 | cupy | 11152 | 830 / 830 / 830 / 830 / 830 | 4.1 / 4.1 / 4.1 / 4.1 / 4.1 |
| vandermonde_lstsq | lstsq | m16384_n64 | numo | — | 出力が崩れた（下記） | — |
| vandermonde_lstsq | lstsq | m16384_n64 | numpy | 1952 | 112 / 112 / 112 / 112 / 112 | — |
| vandermonde_lstsq | qr | m16384_n64 | cumo | 17072 | 1080 / 1080 / 1080 / 1080 / 1080 | 4.1 / 4.1 / 4.1 / 4.1 / 4.1 |
| vandermonde_lstsq | qr | m16384_n64 | cupy | 31112 | 803 / 803 / 803 / 803 / 803 | 4.1 / 4.1 / 4.1 / 4.1 / 4.1 |
| vandermonde_lstsq | qr | m16384_n64 | numo | 1464 | 41 / 113 / 113 / 113 / 113 | — |
| vandermonde_lstsq | qr | m16384_n64 | numpy | 768 | 96 / 96 / 96 / 96 / 96 | — |
| procrustes_pca | svd | N8192_d256 | cumo | 2848 | 1004 / 1004 / 1004 / 1004 / 1004 | 16.3 / 16.5 / 16.5 / 16.5 / 16.5 |
| procrustes_pca | svd | N8192_d256 | cupy | 2168 | 785 / 785 / 785 / 785 / 785 | 8.0 / 8.0 / 8.0 / 8.0 / 8.0 |
| procrustes_pca | svd | N8192_d256 | numo | 280 | 114 / 114 / 114 / 122 / 122 | — |
| procrustes_pca | svd | N8192_d256 | numpy | 296 | 143 / 143 / 143 / 143 / 143 | — |
| procrustes_pca | eigh | N8192_d512 | cumo | 5776 | 934 / 934 / 934 / 934 / 934 | 17.0 / 18.0 / 18.0 / 18.0 / 18.0 |
| procrustes_pca | eigh | N8192_d512 | cupy | 5720 | 764 / 764 / 764 / 764 / 764 | 16.0 / 16.0 / 16.0 / 16.0 / 16.0 |
| procrustes_pca | eigh | N8192_d512 | numo | 872 | 82 / 100 / 100 / 82 / 82 | — |
| procrustes_pca | eigh | N8192_d512 | numpy | 1336 | 128 / 128 / 128 / 128 / 128 | — |
| markov_diffusion | lu_solve | n1024 | cumo | 95008 | 1087 / 1087 / 1087 / 1087 / 1087 | 8.0 / 8.0 / 8.0 / 8.0 / 8.0 |
| markov_diffusion | lu_solve | n1024 | cupy | 80664 | 747 / 747 / 747 / 747 / 748 | 4.0 / 4.0 / 4.0 / 4.0 / 4.0 |
| markov_diffusion | lu_solve | n1024 | numo | 7720 | 67 / 71 / 71 / 74 / 74 | — |
| markov_diffusion | lu_solve | n1024 | numpy | 243184 | 114 / 114 / 114 / 114 / 114 | — |
| markov_diffusion | eig | n256 | cumo | 1120 | 1257 / 1257 / 1257 / 1257 / 1257 | 0.8 / 0.8 / 1.0 / 1.0 / 1.0 |
| markov_diffusion | eig | n256 | cupy | 1136 | 747 / 747 / 747 / 747 / 747 | 0.2 / 0.2 / 0.2 / 0.2 / 0.2 |
| markov_diffusion | eig | n256 | numo | 1584 | 46 / 81 / 81 / 81 / 81 | — |
| markov_diffusion | eig | n256 | numpy | 680 | 111 / 111 / 111 / 111 / 111 | — |
| markov_diffusion | expm | n256 | cumo | 35544 | 882 / 882 / 882 / 882 / 882 | 0.5 / 0.5 / 0.5 / 0.5 / 0.5 |
| markov_diffusion | expm | n256 | cupy | 27864 | 772 / 772 / 772 / 772 / 772 | 0.2 / 0.2 / 0.2 / 0.2 / 0.2 |
| markov_diffusion | expm | n256 | numo | 4552 | 72 / 70 / 70 / 66 / 50 | — |
| markov_diffusion | expm | n256 | numpy | 8320 | 111 / 111 / 111 / 111 / 111 | — |
| kalman_riccati | riccati | n256 | cumo | 112592 | 890 / 890 / 890 / 890 / 890 | 0.9 / 1.1 / 1.4 / 1.4 / 1.4 |
| kalman_riccati | riccati | n256 | cupy | 137976 | 773 / 773 / 773 / 773 / 773 | 0.8 / 0.8 / 0.8 / 0.8 / 0.8 |
| kalman_riccati | riccati | n256 | numo | 9448 | 70 / 87 / 87 / 87 / 87 | — |
| kalman_riccati | riccati | n256 | numpy | 15720 | 161 / 161 / 161 / 161 / 161 | — |
| kkt | solve | n1024 | cumo | 13448 | 915 / 915 / 915 / 915 / 915 | 4.0 / 4.0 / 4.0 / 4.0 / 4.0 |
| kkt | solve | n1024 | cupy | 13832 | 748 / 748 / 748 / 748 / 749 | 4.0 / 4.0 / 4.0 / 4.0 / 4.0 |
| kkt | solve | n1024 | numo | 2224 | 91 / 91 / 91 / 91 / 91 | — |
| kkt | solve | n1024 | numpy | 2584 | 130 / 130 / 130 / 130 / 130 | — |
| kkt | ldl | n1024 | cumo | 1328 | 990 / 990 / 990 / 990 / 990 | 4.0 / 4.0 / 4.0 / 4.0 / 4.0 |
| kkt | ldl | n1024 | numo | 1072 | 87 / 87 / 87 / 87 / 87 | — |
| kkt | ldl | n1024 | numpy | 2016 | 149 / 149 / 149 / 149 / 149 | — |
| hartree_fock | eigh_general | n512 | cumo | 5168 | 928 / 928 / 928 / 928 / 928 | 2.0 / 4.0 / 4.0 / 4.0 / 4.0 |
| hartree_fock | eigh_general | n512 | cupy | 5088 | 760 / 760 / 760 / 760 / 761 | 2.0 / 2.0 / 2.0 / 2.0 / 2.0 |
| hartree_fock | eigh_general | n512 | numo | 1712 | 94 / 68 / 68 / 80 / 72 | — |
| hartree_fock | eigh_general | n512 | numpy | 2360 | 112 / 112 / 112 / 112 / 112 | — |

- Cumoは、12条件すべてでRSSが1 MBの単位まで横ばいだった。プールの使用中バイト数は、最初のN回までに1〜2 MB増えた条件がある（`eigh`の17.0 → 18.0 MB、`eigh_general`の2.0 → 4.0 MBなど）が、そのあとは一定である。Rubyの生存オブジェクトは8N回で約20個増えただけで、記録に使った値そのものである。
- NumoのRSSは数十MBの範囲で上下するが、増え続けてはいない。`lstsq`の行は、SFloatの`lstsq`が出すデバッグ出力（[numo-linalg-alt-notes.md](numo-linalg-alt-notes.md)の1）で表が崩れた。生の出力では41 / 78 / 57 / 53 / （8N）MBで、増え続けてはいない。

## Cumoの起動時の上乗せ

RSSの推移（fp32、256 × 256の行列で、各関数を初めて呼んだ直後）。

| 段階 | Cumo | CuPy |
|---|---|---|
| 起動直後 | 22 MB | 9 MB |
| numo-linalg-alt / numpy + scipy | 26 MB | 60 MB |
| `require 'cumo/narray'` / `import cupy` | 457 MB | 260 MB |
| 最初のカーネル（CUDAのコンテキスト） | 665 MB | 548 MB |
| 最初の`dot`（cuBLAS） | 824 MB | 694 MB |
| 最初の`solve`（cuSOLVER getrf / getrs） | 877 MB | 779 MB |
| 最初の`eigh`（syevd） | 950 MB | 852 MB |
| 最初の`svd`（gesvd） | 1045 MB | 895 MB |

- **差の大部分は`require`の時点で出ている**（+430 MB対 +200 MB）。そのあとの段階ごとの増え方は、両者でほぼ同じである。
- `cumo.so`は115 MBあり、そのうち`.nv_fatbin`（GPUのカーネル）が42 MB。`ldd`で見ると、cuSOLVER、cuDNN、cuBLAS、cuBLASLt、NVRTC、nvJitLink、cuSPARSEを直接リンクしている。
- `require`直後のRSSの主な内訳（`/proc/self/smaps`）: 匿名領域153 MB、libcublasLt 71 MB、nvidiactl 53 MB、libnvrtc 32 MB、libnvJitLink 25 MB。nvidiactlが載っているので、`require`の時点でGPUドライバの初期化も始まっている。CuPyはこれらを必要になったときに読み込む。
- 条件によっては、CumoとCuPyの差がこれより大きい（Fの`eig`のn = 256で1257 MB対747 MB）。この差の内訳は調べていない。

## cumo master `aaa3162b`での段階表（2026-10-03）

#583で、型ごとのカーネルモジュールが分割された。旧ビルド`1c831278`と並べて、段階表を取り直した。RSSはMB、256 × 256の行列。各関数を初めて呼んだ直後の値で、上の表に「最初の要素演算」（`a + a`）と「最初の縮約」（`a.sum`）の2段を足した。3回ずつ回し、3回とも1 MBの単位まで同じ値だった。CuPyは取り直していない。

| 段階 | 旧SFloat | 新SFloat | 旧DFloat | 新DFloat |
|---|---|---|---|---|
| 起動直後 | 22 | 22 | 22 | 22 |
| numo-linalg-alt | 26 | 26 | 26 | 26 |
| `require 'cumo/narray'` | 438 | 439 | 438 | 439 |
| `require 'cumo/linalg'` | 439 | 440 | 439 | 440 |
| 最初のカーネル（`rand`、CUDAのコンテキスト） | 646 | 562 | 656 | 564 |
| 最初の要素演算 | 647 | 562 | 656 | 564 |
| 最初の縮約 | 647 | 607 | 656 | 613 |
| 最初の`dot`（cuBLAS） | 806 | 769 | 794 | 751 |
| 最初の`solve` | 859 | 830 | 822 | 779 |
| 最初の`eigh` | 932 | 903 | 879 | 839 |
| 最初の`svd` | 1027 | 998 | 940 | 900 |

- **最初のカーネルで増える量が減った。** SFloatで +207 MBから +122 MB、DFloatで +217 MBから +124 MBになった。旧ビルドは、ここで型のカーネルを一式読み込んでいた。新ビルドは、最初の縮約でさらに +45〜49 MB増える。縮約まで含めると +167〜173 MBで、旧より40 MBほど少ない。
- `require`の時点の約438 MBは変わらない。最後の`svd`まで回した時点では、旧より29〜40 MB少ない。
