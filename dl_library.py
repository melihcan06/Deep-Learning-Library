import numpy as np


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
        self.deltalar = np.zeros((
                                     biaslar.shape))  # okuldaki sigmoid hatasi dedigimiz deltalar cikti katmanindaki noronlar icin LossF'() * AktivasyonF'()
        # gizli katmanlar icin (E S*w) * AktivasyonF'() = o norona etki eden sonraki katman noronlarinin deltasi * etki eden agirlik


class nn:
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

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid
    def _sigmoid_turev(self, x):  # f_net=sigmoid(net),sigmoid'(net)=f_net*(1-f_net)
        return x * (1 - x)

    # derivative of activation func
    def _aktivasyon_fonk_turev(self, x, ad='sigmoid'):
        if ad == 'sigmoid':
            return self._sigmoid_turev(x)

    # activation func
    def _aktivasyon_fonk(self, x, ad='sigmoid'):
        if ad == 'sigmoid':
            return self._sigmoid(x)

    # sum of output neurons mse
    def _mean_squared_error(self, y, beklenen):
        # y=output beklenen=target
        return np.sum(((beklenen - y) ** 2) / y.shape[1])

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
    def _hata_fonksiyonu(self, y, beklenen, ad='mse'):  # eksana bastirmalik hata,y=bulunan, beklenen=veri setinde olan
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
        return np.array(f_netler_cache)

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
                        turev = f_netler_cache[j] * self.katmanlar[j].deltalar
                        self.katmanlar[j].agirliklar[t] -= np.array(ogrenme_katsayisi * turev).reshape((2,))
                        # w -= learning rate * derivative Error total/derivative w for sgd with momentum = 0

        return 1

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
    def _padding_boyutu_hesaplama(self,girdi_boyutu,cikti_boyutu,filtre_boyutu,kaydirma):#olmasini istedigimiz boyut icin ne kadar padding yapilmali
        # girdi_boyutu=input shape ,cikti_boyutu=output shape,filtre_boyutu=filter shape,kaydirma=stride
        #p=(((o-1)*s)+f-i)/2
        boy_pad = (((cikti_boyutu[0] - 1) * kaydirma[0]) + filtre_boyutu[0] - girdi_boyutu[0]) / 2
        en_pad = (((cikti_boyutu[1] - 1) * kaydirma[1]) + filtre_boyutu[1] - girdi_boyutu[1]) / 2
        return (int(boy_pad), int(en_pad))

    #shape calculation for after convolution
    def _konvolusyon_sonrasi_olusacak_boyut_hesabi(self,goruntu_boyutu,filtre_boyutu,kaydirma,padding=(0,0)):
        #((g-f+2*p)/k)+1=c , ((i-f+2*p)/s)+1=o
        yeni_boy = ((goruntu_boyutu[0] - filtre_boyutu[0] + 2 * padding[0]) / kaydirma[0]) + 1
        yeni_en = ((goruntu_boyutu[1] - filtre_boyutu[1] + 2 * padding[1]) / kaydirma[1]) + 1
        return (int(yeni_boy), int(yeni_en))

    #padding to keep the image the same shape
    def _ayni_boyut_icin_padding(self,grt,filtre_boyutu,kaydirma=(1,1),padding_yontemi='zero'):
        boyut=grt.shape[0],grt.shape[1]
        padding_boyutu=self._padding_boyutu_hesaplama(boyut,boyut,filtre_boyutu,kaydirma)
        return self._padding(grt,padding_boyutu,padding_yontemi)

    #convolution for gray scale image
    def _konvolusyon_gri(self,grt,filtre,kaydirma=(1,1),padding=False,biases=None):#tam kontrol yapilmadi!!!!
        #girdi=input,filtre=filter,kaydirma=stride(tuple(x,y)),if padding is True input shape = output shape
        ksob=self._konvolusyon_sonrasi_olusacak_boyut_hesabi(grt.shape,filtre.shape, kaydirma)
        if padding==True:
            yeni=self._ayni_boyut_icin_padding(grt,filtre.shape,kaydirma,'zero')
        else:
            yeni=np.zeros(ksob,dtype="float32")

        goruntu_boy_bitis = (kaydirma[0]*(ksob[0]-1))+1#yanlis duzelt
        goruntu_en_bitis = (kaydirma[1]*(ksob[1]-1))+1#yanlis duzelt
        yeni_boy_index=0
        for boy in range(0,goruntu_boy_bitis,kaydirma[0]):
            yeni_en_index=0
            for en in range(0,goruntu_en_bitis,kaydirma[1]):
                yeni[yeni_boy_index][yeni_en_index]=np.sum(np.multiply(grt[boy:boy+filtre.shape[0],en:en+filtre.shape[1]],filtre))
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
    def _konvolusyon_rgb(self, grt, filtre, kaydirma=(1,1), padding=False, biases=None):
        b = self._konvolusyon_gri(grt[:, :, 0], filtre, kaydirma, padding=False, biases=None)
        g = self._konvolusyon_gri(grt[:, :, 1], filtre, kaydirma, padding=False, biases=None)
        r = self._konvolusyon_gri(grt[:, :, 2], filtre, kaydirma, padding=False, biases=None)
        return self._rgb_kanallari_birlestir(b,g,r)

    # convolution
    def konvolusyon_islemi(self, grt, filtre, kaydirma=(1,1), padding=False, biases=None):
        if len(grt.shape)==3:
            return self._konvolusyon_rgb(grt, filtre, kaydirma, padding=False, biases=None)
        else:
            return self._konvolusyon_gri(grt, filtre, kaydirma, padding=False, biases=None)


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
        return  np.array(c, dtype=veri_tipi)

    #array to img
    def ciktiyi_goruntuye_cevir(self,grt,olcekleme=True):
        if olcekleme==True:
            if len(grt.shape)==2:
                grt=self.olcekleme2b(grt)
            else:
                grt=self.olcekleme3b(grt)
        return self._sinirla(grt)

ag = nn([2, 2, 2])
# ag.katmanlar=np.array([katman(np.array([[0.15,0.25],[0.2,0.3]]),np.array([[0.35,0.35]])),katman(np.array([[0.4,0.5],[0.45,0.55]]),np.array([[0.6,0.6]]))])
giris = np.array([[0.05, 0.1]])
cikis = np.array([[0.01, 0.99]])
ag.egitim(giris, cikis, 1000, ogrenme_katsayisi=0.5)
print(ag.ileri_yayilim(giris))
