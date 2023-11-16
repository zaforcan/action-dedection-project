# action-dedection-project
spor karşılaşmaları için makine öğrenmesi destekli uygulama (test)




datacollectionfromimages.py uygulasında ile "shot" ve "idle" klasörlerinde bulunan toplam 100 adet görüntü mediapipe kütüphanesi kullanılarak analiz edildi.
bu görüntülerdeki pozisyonlara ait 6 çeşit vücut açısı ölçüsü hesaplandı. 

hesaplanan ölçüler ve sonuçlar tensorflow makine öğrenmesi araçları kullanılarak model oluşturuldu. (google colab'da yapılan işlemlerin sonuçları "my_model" klasörüne kaydedilerek, ana uygulama olan main.py dosyasında tekrar yüklendi.

main.py dosyasının işlevi:

opencv kütüphanesi, verilen görüntüyü frame olarak işliyor. her frame'de mediapipe uygulaması görüntüdeki kişinin vücut açılarını hesaplıyor.
hesaplanan sayılar modelimize aktarılıyor.

eğer eğer "idle" sınıfına ait bir frame varsa: sadece csv dosyasına kaydediliyor 
eğer "shot" sınıfına ait bir frame varsa:  hem ekran görüntüsü kaydediliyor, hem de csv dosyasına kaydediliyor. 




