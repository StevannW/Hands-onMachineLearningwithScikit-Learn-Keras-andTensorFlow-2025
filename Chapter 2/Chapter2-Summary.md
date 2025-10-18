# Ringkasan Bab 2 – End-to-End Machine Learning Project


## Working with Real Data
Data nyata umumnya tidak rapi, memiliki nilai hilang, data ekstrem (outlier), dan hubungan kompleks antar fitur.
Dalam chapter ini, dataset yang digunakan adalah California Housing Prices, yang berisi informasi seperti lokasi (longitude, latitude), usia rumah, jumlah kamar, jumlah rumah tangga, dan pendapatan median wilayah. Dataset ini merupakan versi yang disederhanakan dari data sensus 1990 yang digunakan untuk tujuan pembelajaran.

### Frame the Problem
Langkah pertama adalah memahami konteks bisnis dan merumuskan masalah dalam kerangka Machine Learning. Untuk bisa mendapatkan konteks bisnis dan merumuskan masalahnya, ada beberapa pertanyaan yang harus dijawab, antara lain :
-  Apa tujuan bisnis dari proyek ini?
- Apa output yang diinginkan dari sistem ML?
- Bagaimana hasil model akan digunakan oleh bagian lain dari sistem?
Pada kasus di chapter ini, mode akan memperkirakan harga rumah, sehingga permasalahan dikategorikan sebagai berikut :
- Supervised learning (karena terdapat label, yaitu harga rumah).
- Regression task (target berupa nilai kontinu).
- Batch learning (model dilatih dengan seluruh dataset, bukan online learning).
Setelah kerangka masalah jelas, langkah selanjutnya adalah menentukan ukuran performa (performance measure).

### Select a Performance Measure
Ukuran performa yang digunakan bergantung pada jenis masalah.
Untuk regresi seperti ini, metrik yang paling umum adalah Root Mean Square Error (RMSE) karena:
- Mengukur seberapa jauh prediksi model dari nilai sebenarnya.
- Memberikan penalti lebih besar untuk kesalahan besar (lebih sensitif terhadap outlier).
Alternatif lain adalah Mean Absolute Error (MAE), yang lebih tahan terhadap outlier karena menghitung rata-rata kesalahan absolut.

### Check the Assumptions
Lalu, akan lebih baik ketika membuat list dan mengkoreksi asumsi yang sudah dibuat. Hal ini bisa membantu mencari masalah yang ada nantinya. Contohnya pada chapter 2 ini akan menghasilkan prediksi harga distrik. Jika sistem berikutnya mengonversi harga menjadi kategori seperti murah, mahal, dll maka ketepatan harga yang diprediksi tidak lagi penting. Jika kasusnya seperti ini, maka task yang harus digunakan adalah classification task, bukan regression task.


## Get the Data
Sebelum memulai proyek machine learning, kita harus mempersiapkan workspace dahulu. Berikut ini adalah langkah langkah untuk mempersiapkan workspace.
1. Instalasi python
2. Membuat workspace untuk proyek ML
3. Menginstall modul dan dependensi yang dibutuhkan
4. Memeriksa dan memperbarui pip
5. Menginstall semua modul sekaligus
6. Menyiapkan virtual environment (optional)
7. Eksplorasi Jupyter Notebook