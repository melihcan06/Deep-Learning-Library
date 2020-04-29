import numpy as np

class katman:
    def __init__(self,agirliklar,biaslar):
        self.agirliklar=agirliklar
        self.biaslar=biaslar

class nn:
    def __init__(self,katmanlardaki_noron_sayilari,aktivasyon_fonksiyonlari=None):
        #katmanlardaki_noron_sayilari = [4,3,2] icin 4 giris,1 gizli katman vardir.3 noron gizli katman,2 noron da cikis katmaninda var demektir
        #hata kontrolleri daha sonra yapilacaktir katman sayisi ,0 ve altı girilemez vs...

        self._katmanlardaki_noron_sayilari = katmanlardaki_noron_sayilari

        self._girdi_sayisi=katmanlardaki_noron_sayilari[0]
        self._ara_katman_sayisi=len(katmanlardaki_noron_sayilari)-2
        self._cikti_katmanindaki_noron_sayisi=katmanlardaki_noron_sayilari[-1]

        self.katmanlar=self._katmanlari_olustur()

        if aktivasyon_fonksiyonlari==None:#all degil de none ya da one olacak heralde
            self.aktivasyon_fonksiyonlari=['sigmoid' for i in range(self._ara_katman_sayisi+1)]
        else:
            self.aktivasyon_fonksiyonlari=aktivasyon_fonksiyonlari

    def _katmanlari_olustur(self):
        katmanlar=[]
        for i in range(self._ara_katman_sayisi + 1):
            agirliklar=np.random.random((self._katmanlardaki_noron_sayilari[i], self._katmanlardaki_noron_sayilari[i + 1]))
            biaslar=np.random.random((1, self._katmanlardaki_noron_sayilari[i+1]))
            katmanlar.append(katman(agirliklar,biaslar))
        return np.array(katmanlar)

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def _sigmoid_turev(self,x):
        return x*(1-x)

    def _aktivasyon_fonk(self,ad,x):
        if ad=='sigmoid':
            return self._sigmoid(x)

    def _mean_squared_error(self,y,beklenen):#bi bak dogru mu diye
        return np.sum(((beklenen-y)**2)/y.shape[1])

    def _mean_squared_error_turevi_tekli(self,y,beklenen):#tekli = eo1 eo2 diye ayri ayri donduruluyor demek sigmoidli egitimde kullaniliyor
        return (y-beklenen)#turevi alinca boyle oluyor

    def _hata_fonksiyonu_turevi_tekli(self,y,beklenen,ad='mse'):#formullerde kullanilmak icin turevi alinmis hata
        if ad == 'mse':
            return self._mean_squared_error_turevi_tekli(y,beklenen)

    def _hata_fonksiyonu(self,y,beklenen,ad='mse'):#eksana bastirmalik hata,y=bulunan, beklenen=veri setinde olan
        if ad == 'mse':
            return self._mean_squared_error(y,beklenen)

    def ileri_yayilim2(self,girdi):#eger katman[i]agirliklari bir norondan cikan degil bir sonraki norona gelene gore tutuluyorsa
        f_net = girdi
        f_netler_cache=[girdi]
        for i in range(self._ara_katman_sayisi + 1):
            katm=self.katmanlar[i]
            net=np.dot(f_net,katm.agirliklar.T)+katm.biaslar
            f_net = self._aktivasyon_fonk(self.aktivasyon_fonksiyonlari[i], net)
            f_netler_cache.append(f_net)
        return np.array(f_netler_cache)

    def ileri_yayilim(self,girdi):#eger katman[i]agirliklar[i] i. norondan cikana gore tutuluyorsa
        f_net = girdi
        f_netler_cache=[f_net]
        for i in range(self._ara_katman_sayisi + 1):
            katm = self.katmanlar[i]
            net = np.dot(f_net, katm.agirliklar) + katm.biaslar
            norona_gelenler=[]#toplanmamis hali
            f_net = self._aktivasyon_fonk(self.aktivasyon_fonksiyonlari[i], net)
            f_netler_cache.append(f_net)
        return np.array(f_netler_cache)

    def _sigmoid_fonk_ile_egitim(self,girdi,beklenen,epoch,batch_size=None,hata_fonksiyonu='mse',optimizer='sgd',ogrenme_katsayisi=0.1):
        #girdi boyutu nx... n=veri setinde kac tane ornek varsa ,...= 1 tane ornek=girdi[0]
        for tekrar in range(epoch):#epoch
            for i in range(girdi.shape[0]):#iterasyon
                g = np.reshape(girdi[i], (1, girdi[0].shape[0]))#sgd kullaniliyor
                c = np.reshape(beklenen[i], (1, beklenen[0].shape[0]))#sgd kullaniliyor
                f_netler_cache = self.ileri_yayilim(g)

                #once cikti katmanindan onceki agirliklari guncelleyecegiz
                son_katman_hatalari=self._hata_fonksiyonu_turevi_tekli(f_netler_cache[-1],c)
                sigmoid_turevleri=self._sigmoid_turev(f_netler_cache[-1])

                #kac tane cikti noronu varsa
                for j in range(self._cikti_katmanindaki_noron_sayisi):#self.katmanlar[-1].agirliklar.shape[0]):
                    #print(self.katmanlar[-1].agirliklar[j])
                    y=son_katman_hatalari*sigmoid_turevleri*f_netler_cache[-2]
                    #print(y)
                    #print(son_katman_hatalari[0][1],sigmoid_turevleri[0][1],f_netler_cache[-2][0][1])
                    self.katmanlar[-1].agirliklar[j]-=np.array(ogrenme_katsayisi*y).reshape((2,))
                    #print(self.katmanlar[-1].agirliklar[j])#boyle diyerek 2 agirlikta da 1. ciktinin delta hatasını(okuldaki sigmoid h.) nı kullanmis oldum

                print(self.katmanlar[-1].agirliklar)

                #sondan onceki katmanlarin agirliklarinin guncellenmesi
                #burada dEtotal/dw de farklilik var
                


        return 1

    def egitim(self,girdi,cikti,epoch,batch_size=None,hata_fonksiyonu='mse',optimizer='sgd',ogrenme_katsayisi=0.1):
        self._sigmoid_fonk_ile_egitim(girdi,cikti,epoch,batch_size,hata_fonksiyonu,optimizer,ogrenme_katsayisi)

#ag=nn([2,2,1])
#ag.katmanlar=np.array([katman(np.array([[0.129952,0.570345],[-0.923123,-0.328932]]),np.array([[0.341232,-0.115223]])),katman(np.array([[0.164732],[0.752621]]),np.array([[-0.993423]]))])
#giris=np.array([[0,0]])
#cikis=np.array([[0]])
ag=nn([2,2,2])
#ag.katmanlar=np.array([katman(np.array([[0.15,0.2],[0.25,0.3]]),np.array([[0.35,0.35]])),katman(np.array([[0.4,0.45],[0.5,0.55]]),np.array([[0.6,0.6]]))])
ag.katmanlar=np.array([katman(np.array([[0.15,0.25],[0.2,0.3]]),np.array([[0.35,0.35]])),katman(np.array([[0.4,0.5],[0.45,0.55]]),np.array([[0.6,0.6]]))])
giris=np.array([[0.05,0.1]])
cikis=np.array([[0.01,0.99]])
ag.egitim(giris,cikis,1,ogrenme_katsayisi=0.5)
