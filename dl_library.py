import numpy as np

class aktivasyon():
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid
    def sigmoid_turev(self, x):  # f_net=sigmoid(net),sigmoid'(net)=f_net*(1-f_net)
        return x * (1 - x)

    def relu(self,x):
        if x<0:
            return 0
        else:
            return x

    def relu_turev(self,x):#??????????
        return 0

    def softmax(self,x,y):
        return 1

    def softmax_turev(self,x,y):
        return 1

    def uygula(self,ad='sigmoid',x=None,y=None):
        if ad=='sigmoid':
            return self.sigmoid(x)
        if ad=='relu':
            return self.relu(x)
        if ad=='softmax':
            return self.softmax(x,y)
        else:
            return x

# layer
class katman:
    def __init__(self, agirliklar, biaslar):
        # Actually refers to weights and biases between 2 layers ---> layer[i] and layer[i+1]
        # but deltas is for layer[i+1]

        # weights
        self.agirliklar = agirliklar
        # biases
        self.biaslar = biaslar
        # deltas
        self.deltalar = np.zeros((biaslar.shape))
        # okuldaki sigmoid hatasi dedigimiz deltalar cikti katmanindaki noronlar icin LossF'() * AktivasyonF'()
        # gizli katmanlar icin (E S*w) * AktivasyonF'() = o norona etki eden sonraki katman noronlarinin deltasi * etki eden agirlik

class nn:#tek katmanli nn ekle cnn in sonuna tek katman koyulmak istediginde bu hata veriyor onu ekle!!!!!!
    def __init__(self, katmanlardaki_noron_sayilari, aktivasyon_fonksiyonlari=None):
        # katmanlardaki_noron_sayilari = [4,3,2] icin 4 giris,1 gizli katman vardir.3 noron gizli katman,2 noron da cikis katmaninda var demektir
        # hata kontrolleri daha sonra yapilacaktir katman sayisi ,0 ve altı girilemez vs...

        # list is number of neurons in layers = [4,3,2] for 4 input,1 hidden layer.3 neuron in hidden layer,2 neuron in output layer

        # list is number of neurons in layers
        self._katmanlardaki_noron_sayilari = katmanlardaki_noron_sayilari

        # number of dataset rows
        self._girdi_sayisi = katmanlardaki_noron_sayilari[0]
        # number of hidden layers
        self._ara_katman_sayisi = len(katmanlardaki_noron_sayilari) - 2
        # number of output layer neruons
        self._cikti_katmanindaki_noron_sayisi = katmanlardaki_noron_sayilari[-1]

        self._aktivasyon_islem=aktivasyon()
        self.katmanlar = self._katmanlari_olustur()

        if aktivasyon_fonksiyonlari == None:  # all degil de none ya da one olacak heralde
            self.aktivasyon_fonksiyonlari = ['sigmoid' for i in range(self._ara_katman_sayisi + 1)]
        else:
            self.aktivasyon_fonksiyonlari = aktivasyon_fonksiyonlari

    # create layers
    def _katmanlari_olustur(self):
        katmanlar = []
        for i in range(self._ara_katman_sayisi + 1):
            agirliklar = np.random.random(
                (self._katmanlardaki_noron_sayilari[i], self._katmanlardaki_noron_sayilari[i + 1]))
            biaslar = np.random.random((1, self._katmanlardaki_noron_sayilari[i + 1]))
            katmanlar.append(katman(agirliklar, biaslar))
        return np.array(katmanlar)

    # derivative of activation func
    def _aktivasyon_fonk_turev(self, x, ad='sigmoid'):
        if ad == 'sigmoid':
            return self._aktivasyon_islem.sigmoid_turev(x)

    # activation func
    def _aktivasyon_fonk(self, x, ad='sigmoid'):
        if ad == 'sigmoid':
            return self._aktivasyon_islem.sigmoid(x)

    # sum of output neurons mse
    def _mean_squared_error(self, y, beklenen):
        # y=output beklenen=target
        return np.sum(((beklenen - y) ** 2) / 2)#y.shape[1])

    # derivative of mse for every output neuron
    # ex:f=0.5*(target-output)**2 => f'=output-target
    def _mean_squared_error_turevi_tekli(self, y, beklenen):  # tekli = eo1 eo2 diye ayri ayri donduruluyor
        # y=output beklenen=target
        return (y - beklenen)  # turevi alinca boyle oluyor

    # derivative of loss function for every output neuron
    def _hata_fonksiyonu_turevi_tekli(self, y, beklenen, ad='mse'):  # formullerde kullanilmak icin turevi alinmis hata
        # y=output beklenen=target
        if ad == 'mse':
            return self._mean_squared_error_turevi_tekli(y, beklenen)

    # loss function
    def _hata_fonksiyonu(self, y, beklenen, ad='mse'):  #hata,y=bulunan, beklenen=veri setinde olan
        # value of printing  ,y=output beklenen=target
        if ad == 'mse':
            return self._mean_squared_error(y, beklenen)

    # forward propagation
    def ileri_yayilim(self, girdi):  # eger katman[i]agirliklar[i] i. norondan cikana gore tutuluyorsa
        f_net = girdi
        f_netler_cache = [f_net]
        for i in range(self._ara_katman_sayisi + 1):
            katm = self.katmanlar[i]
            net = np.dot(f_net, katm.agirliklar) + katm.biaslar
            norona_gelenler = []  # toplanmamis hali
            f_net = self._aktivasyon_fonk(net, self.aktivasyon_fonksiyonlari[i])
            f_netler_cache.append(f_net)
        return f_netler_cache

    # train with stochastic gradient descent
    def _sgd_ile_egitme(self, girdi, beklenen, epoch, hata_fonksiyonu='mse', ogrenme_katsayisi=0.1):
        # girdi boyutu nx... n=veri setinde kac tane ornek varsa ,...= 1 tane ornek=girdi[0]
        # input shape nx... n=number of samples in dataset ,...= 1 sample = girdi[0]
        for tekrar in range(epoch):  # epoch
            for i in range(girdi.shape[0]):  # iterasyon
                g = np.reshape(girdi[i], (1, girdi[0].shape[0]))  # sgd kullaniliyor
                c = np.reshape(beklenen[i], (1, beklenen[0].shape[0]))  # sgd using
                f_netler_cache = self.ileri_yayilim(g)

                # cikti katmanindaki noronlarin deltalarini hesaplama
                # calculating output neurons deltas
                son_katman_hatalari = self._hata_fonksiyonu_turevi_tekli(f_netler_cache[-1], c, hata_fonksiyonu)
                aktivasyon_turevleri = self._aktivasyon_fonk_turev(f_netler_cache[-1],
                                                                   self.aktivasyon_fonksiyonlari[-1])
                print(f_netler_cache[-1])
                self.katmanlar[-1].deltalar = aktivasyon_turevleri * son_katman_hatalari

                # gizli katman noronlarinin deltalarini hesaplama
                # calculating hidden neurons deltas
                for j in range(self._ara_katman_sayisi):
                    # delta_zinciri = o norona etki eden sonraki noronlarin deltalarinin agirliklara gore toplami
                    # delta_zinciri = sum of later deltas multiply with wieghts
                    delta_zinciri = np.dot(self.katmanlar[j + 1].deltalar, self.katmanlar[j + 1].agirliklar.T)
                    self.katmanlar[j].deltalar = self._aktivasyon_fonk_turev(f_netler_cache[j + 1]) * delta_zinciri

                # agirliklarin guncellenmesi
                # updating weights
                for j in range(self._ara_katman_sayisi + 1):
                    for t in range(self.katmanlar[j].agirliklar.shape[0]):
                        # turev = agirliga gelen giris ya da girdi * delta
                        # derivative before multiply learning rate
                        turev = f_netler_cache[j][0][t] * self.katmanlar[j].deltalar
                        self.katmanlar[j].agirliklar[t] -= np.array(ogrenme_katsayisi * turev).reshape(self.katmanlar[j].agirliklar[t].shape[0],)
                        # w -= learning rate * derivative Error total/derivative w for sgd with momentum = 0

        return self.katmanlar#[0].deltalar

    # train method
    def egitim(self, girdi, cikti, epoch, batch_size=None, hata_fonksiyonu='mse', optimizer={}, ogrenme_katsayisi=0.1):
        optimizer['ad'] = 'sgd'
        optimizer['momentum'] = 0
        self._sgd_ile_egitme(girdi, cikti, epoch, hata_fonksiyonu, ogrenme_katsayisi)

class konvolusyon:
    def _zero_padding(self,grt,padding_boyutu):
        #padding_boyutu is padding shape.(y pad , x pad)
        (y_pad, x_pad)=padding_boyutu
        return np.pad(grt, ((y_pad, y_pad), (x_pad, x_pad), (0, 0)), 'constant', constant_values=0)

    def _padding(self,grt,padding_boyutu,padding_yontemi='zero'):
        #padding_boyutu=padding shape
        #padding_yontemi=padding technique
        if padding_yontemi=='zero':
            return self._zero_padding(grt,padding_boyutu)
        return 1

    # how we do padding for the shape we want
    def padding_boyutu_hesaplama(self,girdi_boyutu,cikti_boyutu,filtre_boyutu,kaydirma):#olmasini istedigimiz boyut icin ne kadar padding yapilmali
        # girdi_boyutu=input shape ,cikti_boyutu=output shape,filtre_boyutu=filter shape,kaydirma=stride
        #p=(((o-1)*s)+f-i)/2
        boy_pad = (((cikti_boyutu[0] - 1) * kaydirma[0]) + filtre_boyutu[0] - girdi_boyutu[0]) / 2
        en_pad = (((cikti_boyutu[1] - 1) * kaydirma[1]) + filtre_boyutu[1] - girdi_boyutu[1]) / 2
        return (int(boy_pad), int(en_pad))

    #shape calculation for after convolution
    def konvolusyon_sonrasi_olusacak_boyut_hesabi(self,goruntu_boyutu,filtre_boyutu,kaydirma,padding=(0,0)):
        #((g-f+2*p)/k)+1=c , ((i-f+2*p)/s)+1=o
        yeni_boy = ((goruntu_boyutu[0] - filtre_boyutu[0] + 2 * padding[0]) / kaydirma[0]) + 1
        yeni_en = ((goruntu_boyutu[1] - filtre_boyutu[1] + 2 * padding[1]) / kaydirma[1]) + 1
        return (int(yeni_boy), int(yeni_en))

    #padding to keep the image the same shape
    def ayni_boyut_icin_padding(self,grt,filtre_boyutu,kaydirma=(1,1),padding_yontemi='zero'):
        boyut=grt.shape[0],grt.shape[1]
        padding_boyutu=self.padding_boyutu_hesaplama(boyut,boyut,filtre_boyutu,kaydirma)
        return self._padding(grt,padding_boyutu,padding_yontemi)

    #convolution for gray scale image
    def _konvolusyon_gri(self,grt,filtre,kaydirma=(1,1),padding=False,aktivasyon_fonksiyonu='relu',biases=None):#tam kontrol yapilmadi!!!!
        aktv=aktivasyon()
        #girdi=input,filtre=filter,kaydirma=stride(tuple(x,y)),if padding is True input shape = output shape
        ksob=self.konvolusyon_sonrasi_olusacak_boyut_hesabi(grt.shape,filtre.shape, kaydirma)
        if padding==True:
            yeni=self.ayni_boyut_icin_padding(grt,filtre.shape,kaydirma,'zero')
        else:
            yeni=np.zeros(ksob,dtype="float32")

        goruntu_boy_bitis = (kaydirma[0]*(ksob[0]-1))+1#yanlis duzelt
        goruntu_en_bitis = (kaydirma[1]*(ksob[1]-1))+1#yanlis duzelt
        yeni_boy_index=0
        for boy in range(0,goruntu_boy_bitis,kaydirma[0]):
            yeni_en_index=0
            for en in range(0,goruntu_en_bitis,kaydirma[1]):
                deger=np.sum(np.multiply(grt[boy:boy+filtre.shape[0],en:en+filtre.shape[1]],filtre))
                yeni[yeni_boy_index][yeni_en_index]=aktv.uygula(aktivasyon_fonksiyonu,deger)
                yeni_en_index+=1
            yeni_boy_index+=1

        return yeni

    #concatenating r g b channels
    def _rgb_kanallari_birlestir(self,b,g,r,veri_tipi="float32"):
        yeni = np.zeros((b.shape[0], b.shape[1], 3), dtype=veri_tipi)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                yeni[i][j][0] = b[i][j]
                yeni[i][j][1] = g[i][j]
                yeni[i][j][2] = r[i][j]
        return yeni

    # convolution for rgb image
    def _konvolusyon_rgb(self, grt, filtre, kaydirma=(1,1), padding=False,aktivasyon_fonksiyonu='relu', biases=None):
        b = self._konvolusyon_gri(grt[:, :, 0], filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)
        g = self._konvolusyon_gri(grt[:, :, 1], filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)
        r = self._konvolusyon_gri(grt[:, :, 2], filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)
        return self._rgb_kanallari_birlestir(b,g,r)

    # convolution
    def konvolusyon_islemi(self, grt, filtre, kaydirma=(1,1), padding=False, aktivasyon_fonksiyonu='relu', biases=None):
        if len(grt.shape)==3:
            return self._konvolusyon_rgb(grt, filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)
        else:
            return self._konvolusyon_gri(grt, filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)

    #alt_sinir=new value of min
    #ust_sinir new value of max
    # normalization 2d
    def olcekleme2b(self,girdi, alt_sinir=0, ust_sinir=255):
        eb = np.max(girdi)
        ek = np.min(girdi)
        ebek = eb - ek
        sinir_fark = ust_sinir - alt_sinir

        for i in range(girdi.shape[0]):
            for j in range(girdi.shape[1]):
                girdi[i][j] = ((girdi[i][j] - ek) / ebek) * (sinir_fark) + alt_sinir

        return girdi

    # normalization 3d
    def olcekleme3b(self,girdi, alt_sinir=0, ust_sinir=255):
        eb = np.max(girdi)
        ek = np.min(girdi)
        ebek = eb - ek
        sinir_fark = ust_sinir - alt_sinir

        for i in range(girdi.shape[0]):
            for j in range(girdi.shape[1]):
                girdi[i][j][0] = ((girdi[i][j][0] - ek) / ebek) * (sinir_fark) + alt_sinir
                girdi[i][j][1] = ((girdi[i][j][1] - ek) / ebek) * (sinir_fark) + alt_sinir
                girdi[i][j][2] = ((girdi[i][j][2] - ek) / ebek) * (sinir_fark) + alt_sinir

        return girdi

    # 2 threshold values are used first and their numbers are taken.Used when converting array to img
    def _sinirla(self,a,alt=0,ust=255,veri_tipi="uint8"):
        c = np.clip(a, alt, ust)
        return np.array(c, dtype=veri_tipi)

    #array to img
    def ciktiyi_goruntuye_cevir(self,grt,olcekleme=True):
        if olcekleme==True:
            if len(grt.shape)==2:
                grt=self.olcekleme2b(grt)
            else:
                grt=self.olcekleme3b(grt)
        return self._sinirla(grt)

class pooling:
    # shape calculation for after convolution #convda da var
    def konvolusyon_sonrasi_olusacak_boyut_hesabi(self, goruntu_boyutu, filtre_boyutu, kaydirma, padding=(0, 0)):
        # ((g-f+2*p)/k)+1=c , ((i-f+2*p)/s)+1=o
        yeni_boy = ((goruntu_boyutu[0] - filtre_boyutu[0] + 2 * padding[0]) / kaydirma[0]) + 1
        yeni_en = ((goruntu_boyutu[1] - filtre_boyutu[1] + 2 * padding[1]) / kaydirma[1]) + 1
        return (int(yeni_boy), int(yeni_en))

    # concatenating r g b channels #convda da var
    def _rgb_kanallari_birlestir(self, b, g, r, veri_tipi="float32"):
        yeni = np.zeros((b.shape[0], b.shape[1], 3), dtype=veri_tipi)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                yeni[i][j][0] = b[i][j]
                yeni[i][j][1] = g[i][j]
                yeni[i][j][2] = r[i][j]
        return yeni

    def _pooling_gri(self,grt,filtre_boyutu,kaydirma,yontem='max'):
        ksob = self.konvolusyon_sonrasi_olusacak_boyut_hesabi(grt.shape, filtre_boyutu, kaydirma)
        yeni = np.zeros(ksob, dtype="float32")
        goruntu_boy_bitis = (kaydirma[0] * (ksob[0] - 1)) + 1  # yanlis duzelt
        goruntu_en_bitis = (kaydirma[1] * (ksob[1] - 1)) + 1  # yanlis duzelt
        yeni_boy_index = 0
        if yontem=='max':
            for boy in range(0, goruntu_boy_bitis, kaydirma[0]):
                yeni_en_index = 0
                for en in range(0, goruntu_en_bitis, kaydirma[1]):
                    yeni[yeni_boy_index][yeni_en_index] = np.max(grt[boy:boy + filtre_boyutu[0], en:en + filtre_boyutu[1]])
                    yeni_en_index += 1
                yeni_boy_index += 1
        else:#yontem==mean or average
            for boy in range(0, goruntu_boy_bitis, kaydirma[0]):
                yeni_en_index = 0
                for en in range(0, goruntu_en_bitis, kaydirma[1]):
                    yeni[yeni_boy_index][yeni_en_index] = np.mean(grt[boy:boy + filtre_boyutu[0], en:en + filtre_boyutu[1]])
                    yeni_en_index += 1
                yeni_boy_index += 1
        return yeni

    def _pooling_rgb(self, grt, filtre_boyutu, kaydirma, yontem='max'):
        b = self._pooling_gri(grt[:, :, 0], filtre_boyutu, kaydirma, yontem)
        g = self._pooling_gri(grt[:, :, 1], filtre_boyutu, kaydirma, yontem)
        r = self._pooling_gri(grt[:, :, 2], filtre_boyutu, kaydirma, yontem)
        return self._rgb_kanallari_birlestir(b, g, r)

    def pooling_islemi(self, grt, filtre_boyutu, kaydirma, yontem='max'):
        if len(grt.shape)==3:
            return self._pooling_rgb(grt, filtre_boyutu, kaydirma, yontem)
        else:
            return self._pooling_gri(grt, filtre_boyutu, kaydirma, yontem)

class filtreler:
    def __init__(self, sayi, boyut, bias_sayisi=None):
        self.agirliklar = np.random.random((sayi,boyut[0],boyut[1]))
        #self.biaslar = np.random.random((sayi,bias_sayisi))
        #self.deltalar = np.zeros((self.biaslar.shape))

class cnn:
    def __init__(self):
        self._konvolusyon_islemleri=konvolusyon()
        self._pooling_islemleri=pooling()
        self._katmanlar=[]
        self._konv_katmani_num = 1
        self._pooling_katmani_num = 1
        self._ysa_katmani_num = 1

    def _tensoru_topla(self,x):
        y=x[0,:,:]
        for i in range(1,x.shape[0]):
            y=np.add(y,x[i,:,:])
        return y

    def _rgb_topla(self,x):
        y=x[:,:,0]
        for i in range(1,3):
            y=np.add(y,x[:,:,i])
        return y

    def _duzlestirme(self,x):
        return np.reshape(x,(1,(x.shape[0]*x.shape[1])))

    def konvolusyon_katmani_ekle(self,filtre_sayisi,filtre_boyutu,kaydirma,aktivasyon_fonksiyonu,padding=False):
        konv_k={'ad':'konv'+str(self._konv_katmani_num),'filtre_sayisi':filtre_sayisi,'filtre_boyutu':filtre_boyutu,'kaydirma':kaydirma,
                'aktivasyon_fonksiyonu':aktivasyon_fonksiyonu,'padding':padding}
        #padding == False normal conv shape decrease, padding == True after konv input shape == output shape
        self._konv_katmani_num+=1
        self._katmanlar.append(konv_k)

    def pooling_katmani_ekle(self,filtre_boyutu,kaydirma,yontem='max'):
        pool_k={'ad':'pool'+str(self._pooling_katmani_num),'filtre_boyutu':filtre_boyutu,'kaydirma':kaydirma,'yontem':yontem}
        self._pooling_katmani_num+=1
        self._katmanlar.append(pool_k)

    #flatten layer
    def duzlestirme_katmani_ekle(self):
        duz_k={'ad':'duzlestirme'}
        self._katmanlar.append(duz_k)

    def ysa_katmani_ekle(self,noron_sayisi,aktivasyon_fonksiyonu):
        ysa_k={'ad':'ysa_katman'+str(self._ysa_katmani_num),'noron_sayisi':noron_sayisi,'aktivasyon_fonksiyonu':aktivasyon_fonksiyonu}
        self._ysa_katmani_num+=1
        self._katmanlar.append(ysa_k)

    def _girdi_tensoru_boyutu_hesapla(self,girdi_boyutu):
        cikti_boyutu=girdi_boyutu
        for i in range(len(self._katmanlar)):
            if self._katmanlar[i]['ad'].count('konv') == 1:
                if self._katmanlar[i]['padding']==False:
                    cikti_boyutu=self._konvolusyon_islemleri.konvolusyon_sonrasi_olusacak_boyut_hesabi(
                        cikti_boyutu,self._katmanlar[i]['filtre_boyutu'],self._katmanlar[i]['kaydirma'])
            elif self._katmanlar[i]['ad'].count('pool') == 1:
                cikti_boyutu = self._konvolusyon_islemleri.konvolusyon_sonrasi_olusacak_boyut_hesabi(
                    cikti_boyutu,self._katmanlar[i]['filtre_boyutu'],self._katmanlar[i]['kaydirma'])
        return cikti_boyutu

    def egit(self, girdi, cikti, epoch, batch_size=None, hata_fonksiyonu='mse', optimizer={}, ogrenme_katsayisi=0.1):
        # ysa olusturuldu
        nn_girdi_boyutu = self._girdi_tensoru_boyutu_hesapla((girdi.shape[1],girdi.shape[2]))
        ysa_noron_sayilari = [nn_girdi_boyutu[0]*nn_girdi_boyutu[1]]
        ysa_aktv_fonklari = []
        for i in range(len(self._katmanlar)):
            if self._katmanlar[i]['ad'].count('ysa_katman') == 1:
                ysa_noron_sayilari.append(self._katmanlar[i]['noron_sayisi'])
                ysa_aktv_fonklari.append(self._katmanlar[i]['aktivasyon_fonksiyonu'])
        self._ysa = nn(ysa_noron_sayilari, ysa_aktv_fonklari)

        # filtreler olusturuldu
        self._cnn_filtreler = []
        for i in range(len(self._katmanlar)):
            if self._katmanlar[i]['ad'].count('konv') == 1:
                self._cnn_filtreler.append(
                    filtreler(self._katmanlar[i]['filtre_sayisi'], self._katmanlar[i]['filtre_boyutu']))

        for tekrar in range(epoch):
            x=girdi[tekrar,:,:,:]
            y=np.array(cikti[0][tekrar]).reshape((1,1))#1xN lik vektor
            # ileri yayilim
            konv_cache = []  # konv_cache[katman][filtre_sayisi] ordaki conv sonucunu verir
            konv_cache.append([x])
            konv_katman_sayisi = 0
            for i in range(len(self._katmanlar)):
                if self._katmanlar[i]['ad'].count('konv') == 1:
                    print("girdi")
                    konv_alt_cache = []
                    #butun filtreler uygunaliyor
                    for f in range(self._cnn_filtreler[konv_katman_sayisi].agirliklar.shape[0]):
                        eleman = self._konvolusyon_islemleri.konvolusyon_islemi(x, self._filtreyi_dondur(
                            self._cnn_filtreler[konv_katman_sayisi].agirliklar[f])
                                                                           , self._katmanlar[i]['kaydirma'],
                                                                           self._katmanlar[i]['padding'],
                                                                           self._katmanlar[i]['aktivasyon_fonksiyonu'])
                        #rgb ciktiyi gray scale e donusturur
                        if konv_katman_sayisi == 0 and len(x.shape) == 3:
                            eleman = self._rgb_topla(eleman)
                        konv_alt_cache.append(eleman)

                    konv_cache.append(konv_alt_cache)
                    #cikti tensoru (filtre sayisi kadar) yani ozellik haritalari toplaniyor
                    x = self._tensoru_topla(np.array(konv_alt_cache).reshape((len(konv_alt_cache),eleman.shape[0],eleman.shape[1])))
                    konv_katman_sayisi += 1
                elif self._katmanlar[i]['ad'].count('pool') == 1:
                    x = self._pooling_islemleri.pooling_islemi(x, self._katmanlar[i]['filtre_boyutu'],
                                                               self._katmanlar[i]['kaydirma'],
                                                               self._katmanlar[i]['yontem'])
                    # return self._konvolusyon_islemleri.ciktiyi_goruntuye_cevir(y)

            #duzlestirme
            x=self._duzlestirme(x)

            # geri yayilim
            self._ysa.egitim(x,y,1,hata_fonksiyonu=hata_fonksiyonu,optimizer=optimizer,ogrenme_katsayisi=ogrenme_katsayisi)
            print(self._ysa.katmanlar)


    def _filtreyi_dondur(self,x):
        return np.rot90(np.rot90(x))

    def tamin_yap(self,x):
        return 1

    def katmanlari_bas(self):
        for i in self._katmanlar:
            print(i)


cnn1=cnn()
#cnn1.cnn_giris_katmani_ekle((100,100))
cnn1.konvolusyon_katmani_ekle(2,(3,3),(1,1),'relu')
#cnn1.konvolusyon_katmani_ekle(2,(3,3),(1,1),'relu')
cnn1.pooling_katmani_ekle((3,3),(3,3))
cnn1.duzlestirme_katmani_ekle()
cnn1.ysa_katmani_ekle(2,'sigmoid')
cnn1.katmanlari_bas()
import cv2
a=cv2.resize(cv2.imread("commandos.jpg"),(50,50))
a=np.expand_dims(a,axis=0)
b=np.array([[1]])
cnn1.egit(a,b,1)
"""a=cv2.imread("commandos.jpg")
b=cnn1.deneme_methodu(a)
cv2.imshow("1",a)
cv2.imshow("2",b)
cv2.waitKey(0)
print(1)"""

"""a=np.array([[1,2,3,4,5],
            [6,7,8,9,10],
            [1,2,3,4,5],
            [6,7,8,9,10],
            [1,2,3,4,5],
            [6,7,8,9,10]])
b=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
cnn1=konvolusyon()
print(cnn1.konvolusyon_gri(a,b,(1,1)))"""

"""ag = nn([2, 2, 2])
ag.katmanlar=np.array([katman(np.array([[0.15,0.25],[0.2,0.3]]),np.array([[0.35,0.35]])),katman(np.array([[0.4,0.5],[0.45,0.55]]),np.array([[0.6,0.6]]))])
giris = np.array([[0.05, 0.1]])
cikis = np.array([[0.01, 0.99]])
ag.egitim(giris, cikis, 1, ogrenme_katsayisi=0.5)
print(ag.ileri_yayilim(giris))"""

"""ag = nn([2, 3,4,5, 2])
giris = np.array([[0.05, 0.1]])
cikis = np.array([[0.01, 0.99]])
ag.egitim(giris, cikis, 1000, ogrenme_katsayisi=0.5)
print(ag.ileri_yayilim(giris))"""
