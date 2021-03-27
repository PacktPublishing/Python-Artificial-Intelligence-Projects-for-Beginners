# Tugas pertemuan 3
1. Carilah dataset yang berbeda dan taruh di dalam folder dataset
2. Buatlah file dengan namaNPM.py di folder Chapter02 yang berisi tiga fungsi menggunakan metode random forest: preparation, training, testing
3. fungsi preparation berisi pengambilan data set dan persiapan atribut/fitur dan mengembalikan dataframe testing,training, dan full
4. fungsi training mengembalikan berupa model decission tree dari inputan data training dan label training
5. fungsi testing mengembalikan label prediksi dari inputan data testing 
6. buat fungsi testing tambahkan di bagian bawah file test_app.py, fungsi test dengan nama : test_03_nama_npm, yang berisi untuk melakukan testing tiga fungsi yang sudah anda buat dengan dataset anda sendiri
7. Dikumpulkan dengan cara merge request ke branch 2021b untuk kelas b, 2021c untuk kelas c, maksimal hari jumat jam 11.00 wib, judul diisi nama npm, deskripsi diisi penjelasan jelas bagaimana kelas kelas dan method tersebut bekerja. dijelaskan secara poin

# Kriteria Penilaian
1. Fork hanya boleh sekali seumur hidup jika anda melakukan fork lagi maka diskon nilai 50 persen
2. Selain di merge request ke branch 2021c atau 2021b maka merge akan ditolak
3. Jika tidak pass hijau dari travis CI, maka merge ditolak
4. Jika konflik, maka merge ditolak
5. Jika melakukan perubahan selain tiga file yang dikerjakan, maka merge ditolak
6. Jika menghapus atau merubah pekerjaan orang lain, maka merge ditolak
7. Merge ditolak, maka nilai nol
8. Total nilai akan dikalikan dengan persentasi code coverage dari file(bukan total) yang dibuat hasil dari Travis CI