import numpy as np

#layer
class katman:
    def __init__(self,agirliklar,biaslar):
        #Actually refers to weights and biases between 2 layers ---> layer[i] and layer[i+1]
        #but deltas is for layer[i+1]

        #weights
        self.agirliklar=agirliklar
        #biases
        self.biaslar=biaslar
        #deltas
        self.deltalar=np.zeros((biaslar.shape))#okuldaki sigmoid hatasi dedigimiz deltalar cikti katmanindaki noronlar icin LossF'() * AktivasyonF'()
        #gizli katmanlar icin (E S*w) * AktivasyonF'() = o norona etki eden sonraki katman noronlarinin deltasi * etki eden agirlik

class nn:
    def __init__(self,katmanlardaki_noron_sayilari,aktivasyon_fonksiyonlari=None):
        #katmanlardaki_noron_sayilari = [4,3,2] icin 4 giris,1 gizli katman vardir.3 noron gizli katman,2 noron da cikis katmaninda var demektir
        #hata kontrolleri daha sonra yapilacaktir katman sayisi ,0 ve altı girilemez vs...
        
        #list is number of neurons in layers = [4,3,2] for 4 input,1 hidden layer.3 neuron in hidden layer,2 neuron in output layer  
        
        #list is number of neurons in layers
        self._katmanlardaki_noron_sayilari = katmanlardaki_noron_sayilari

        #number of dataset rows
        self._girdi_sayisi=katmanlardaki_noron_sayilari[0]
        # number of hidden layers
        self._ara_katman_sayisi=len(katmanlardaki_noron_sayilari)-2
        # number of output layer's neruons
        self._cikti_katmanindaki_noron_sayisi=katmanlardaki_noron_sayilari[-1]

        self.katmanlar=self._katmanlari_olustur()

        if aktivasyon_fonksiyonlari==None:#all degil de none ya da one olacak heralde
            self.aktivasyon_fonksiyonlari=['sigmoid' for i in range(self._ara_katman_sayisi+1)]
        else:
            self.aktivasyon_fonksiyonlari=aktivasyon_fonksiyonlari

    #create layers
    def _katmanlari_olustur(self):
        katmanlar=[]
        for i in range(self._ara_katman_sayisi + 1):
            agirliklar=np.random.random((self._katmanlardaki_noron_sayilari[i], self._katmanlardaki_noron_sayilari[i + 1]))
            biaslar=np.random.random((1, self._katmanlardaki_noron_sayilari[i+1]))
            katmanlar.append(katman(agirliklar,biaslar))
        return np.array(katmanlar)

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

    #derivative of sigmoid
    def _sigmoid_turev(self,x):#f_net=sigmoid(net),sigmoid'(net)=f_net*(1-f_net)
        return x*(1-x)

    #derivative of activation func
    def _aktivasyon_fonk_turev(self,x,ad='sigmoid'):
        if ad=='sigmoid':
            return self._sigmoid_turev(x)

    #activation func
    def _aktivasyon_fonk(self,x,ad='sigmoid'):
        if ad=='sigmoid':
            return self._sigmoid(x)

    #sum of output neurons mse
    def _mean_squared_error(self,y,beklenen):
        return np.sum(((beklenen-y)**2)/y.shape[1])

    #derivative of mse for every output neuron
    def _mean_squared_error_turevi_tekli(self,y,beklenen):#tekli = eo1 eo2 diye ayri ayri donduruluyor
        return (y-beklenen)#turevi alinca boyle oluyor

    #derivative of loss function for every output neuron
    #ex:f=0.5*(target-output)**2 => f'=output-target
    def _hata_fonksiyonu_turevi_tekli(self,y,beklenen,ad='mse'):#formullerde kullanilmak icin turevi alinmis hata
        if ad == 'mse':
            return self._mean_squared_error_turevi_tekli(y,beklenen)

    #loss function
    def _hata_fonksiyonu(self,y,beklenen,ad='mse'):#eksana bastirmalik hata,y=bulunan, beklenen=veri setinde olan
        #value of printing  ,y=output beklenen=target
        if ad == 'mse':
            return self._mean_squared_error(y,beklenen)

    #forward propagation
    def ileri_yayilim(self,girdi):#eger katman[i]agirliklar[i] i. norondan cikana gore tutuluyorsa
        f_net = girdi
        f_netler_cache=[f_net]
        for i in range(self._ara_katman_sayisi + 1):
            katm = self.katmanlar[i]
            net = np.dot(f_net, katm.agirliklar) + katm.biaslar
            norona_gelenler=[]#toplanmamis hali
            f_net = self._aktivasyon_fonk(net,self.aktivasyon_fonksiyonlari[i])
            f_netler_cache.append(f_net)
        return np.array(f_netler_cache)

    #train with stochastic gradient descent
    def _sgd_ile_egitme(self,girdi,beklenen,epoch,hata_fonksiyonu='mse',ogrenme_katsayisi=0.1):
        #girdi boyutu nx... n=veri setinde kac tane ornek varsa ,...= 1 tane ornek=girdi[0]
        #input shape nx... n=number of samples in dataset ,...= 1 sample = girdi[0]
        for tekrar in range(epoch):#epoch
            for i in range(girdi.shape[0]):#iterasyon
                g = np.reshape(girdi[i], (1, girdi[0].shape[0]))#sgd kullaniliyor
                c = np.reshape(beklenen[i], (1, beklenen[0].shape[0]))#sgd using
                f_netler_cache = self.ileri_yayilim(g)

                #cikti katmanindaki noronlarin deltalarini hesaplama
                # calculating output neurons deltas
                son_katman_hatalari=self._hata_fonksiyonu_turevi_tekli(f_netler_cache[-1],c,hata_fonksiyonu)
                aktivasyon_turevleri=self._aktivasyon_fonk_turev(f_netler_cache[-1],self.aktivasyon_fonksiyonlari[-1])
                self.katmanlar[-1].deltalar=aktivasyon_turevleri*son_katman_hatalari

                #gizli katman noronlarinin deltalarini hesaplama
                #calculating hidden neurons deltas
                for j in range(self._ara_katman_sayisi):
                    #delta_zinciri = o norona etki eden sonraki noronlarin deltalarinin agirliklara gore toplami
                    #delta_zinciri = sum of later deltas multiply with wieghts
                    delta_zinciri=np.dot(self.katmanlar[j+1].deltalar,self.katmanlar[j+1].agirliklar.T)
                    self.katmanlar[j].deltalar=self._aktivasyon_fonk_turev(f_netler_cache[j+1])*delta_zinciri

                #agirliklarin guncellenmesi
                #updating weights
                for j in range(self._ara_katman_sayisi+1):
                    for t in range(self.katmanlar[j].agirliklar.shape[0]):
                        #turev = agirliga gelen giris ya da girdi * delta
                        #derivative before with multiply learning rate
                        turev=f_netler_cache[j]*self.katmanlar[j].deltalar
                        self.katmanlar[j].agirliklar[t]-=np.array(ogrenme_katsayisi*turev).reshape((2,))

        return 1

    #train method
    def egitim(self,girdi,cikti,epoch,batch_size=None,hata_fonksiyonu='mse',optimizer={},ogrenme_katsayisi=0.1):
        optimizer['ad']='sgd'
        optimizer['momentum'] = 0
        self._sgd_ile_egitme(girdi, cikti, epoch, hata_fonksiyonu, ogrenme_katsayisi)

ag=nn([2,2,2])
#ag.katmanlar=np.array([katman(np.array([[0.15,0.25],[0.2,0.3]]),np.array([[0.35,0.35]])),katman(np.array([[0.4,0.5],[0.45,0.55]]),np.array([[0.6,0.6]]))])
giris=np.array([[0.05,0.1]])
cikis=np.array([[0.01,0.99]])
ag.egitim(giris,cikis,1000,ogrenme_katsayisi=0.5)
print(ag.ileri_yayilim(giris))
