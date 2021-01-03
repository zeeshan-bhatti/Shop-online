from django import template
from login.models import Customer
from home.models import Product,Category,SubCategory
from products.models import Orders
import datetime
from django.db.models import Count,Aggregate,Sum
# supress warnings
import warnings
warnings.filterwarnings('ignore')

#Importing Libraries
import numpy as np
import pandas as pd

# For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# To Scale our data
from sklearn.preprocessing import scale

# To perform KMeans clustering 
from sklearn.cluster import KMeans

# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan




register=template.Library()









@register.filter(name='is_in_cart')
def is_in_cart(product,cart):
    keys=cart.keys()
    for id in keys:
        if int(id) == product.id:
            return True

    return False

@register.filter(name='cart_quantity')
def cart_quantity(product,cart):
    keys=cart.keys()
    for id in keys:
        if int(id) == product.id:
            return cart.get(id)

    return 0

@register.filter(name='total_price')
def total_price(product,cart):
    if product.discount:
        price=product.price - (product.price*product.discount/100)
        return price * cart_quantity(product,cart)
    else: 
        return product.price * cart_quantity(product,cart)

@register.filter(name='grand_total')
def grand_total(products,cart):
    sum=0
    for prod in products:
        sum +=total_price(prod,cart)

    return sum

@register.filter(name='multiply')
def multiply(num1,num2):
    return num1*num2


@register.filter(name='calculate_discount')
def calculate_discount(product):
    return product.price - (product.price*product.discount/100)

@register.filter(name='get_username')
def get_username(customer):
    if customer:
        cust=Customer.objects.filter(id=customer)
        return cust[0].name
    return None

@register.filter(name='get_email')
def get_email(customer):
    if customer:
        cust=Customer.objects.filter(id=customer)
        return cust[0].email
    return None

@register.filter(name='get_gender')
def get_email(customer):
    if customer:
        cust=Customer.objects.filter(id=customer)
        if cust[0].gender=='M':
            return 'Male'
        else:
            return 'Female'
    return None

@register.filter(name='get_age')
def get_email(customer):
    if customer:
        cust=Customer.objects.filter(id=customer)
        return cust[0].age
    return None
    

@register.filter(name='match')
def match(item,cat):
    if cat:
        if item.id==int(cat):
            if cat[0]!="0":
                return True
        else:
            return False
    else:
        return False


@register.filter(name='sub_match')
def sub_match(item,cat):
    if cat:
        if item.id==int(cat):
            if cat[0]=="0":
                return True
        else:
            return False
    else:
        return False



@register.filter(name='check_active')
def check_active(cat):
    if cat=='':
        return True
    else:
        return False

@register.filter(name='products')
def products(name):
    return Product.objects.filter()

@register.filter(name='male_length')
def male_length(name):
     p=Customer.objects.filter(gender='M')
     
     return len(p)

@register.filter(name='female_length')
def female_length(name):
     p=Customer.objects.filter(gender='F')
    
     return len(p)
    
@register.filter(name='online_payment_length')
def online_payment_length(name):
     p=Orders.objects.filter(payment_method='online')
     
     return len(p)

@register.filter(name='cod_payment_length')
def cod_payment_length(name):
     p=Orders.objects.filter(payment_method='cod')
     
     return len(p)

@register.filter(name='deliverd_status_length')
def deliverd_status_length(name):
     p=Orders.objects.filter(status=True)
     
     return len(p)

@register.filter(name='undeliverd_status_length')
def undeliverd_status_length(name):
     p=Orders.objects.filter(status=False)
     return len(p)


@register.filter(name='month_orders')
def month_orders(month):
     p=Orders.objects.filter(order_date__month=month).filter(order_date__year=datetime. datetime. now(). year)
     return len(p)


@register.filter(name='orders')
def orders(name):
     ords=Orders.objects.values(name).order_by(name).annotate(dcount=Count(name))
     return ords.order_by('-dcount')


@register.filter(name='total_sales')
def total_sales(name):
     ords=Orders.objects.aggregate(Sum('price'))
     return ords['price__sum']


@register.filter(name='total_quantity')
def total_quantity(id):
     ords=Orders.objects.filter(product=id)
     ords=ords.aggregate(Sum('quantity'))
     return ords['quantity__sum']


@register.filter(name='get_price')
def get_price(id,attribute):
    if attribute=="product":
        ords=Orders.objects.filter(product=id)
        ords=ords.aggregate(Sum('price'))
        return ords['price__sum']

    if attribute=="customer":
        ords=Orders.objects.filter(customer=id)
        ords=ords.aggregate(Sum('price'))
        return ords['price__sum']
    


@register.filter(name='month_customers')
def month_customers(month):
     p=Customer.objects.filter(register_date__month=month).filter(register_date__year=datetime. datetime. now(). year)
     return len(p)


@register.filter(name='category')
def category(name):
     p=Category.objects.all()
     return p



@register.filter(name='category_orders')
def category_orders(id):
     p=Orders.objects.filter(product__category=id)
     return len(p)


@register.filter(name='category_products')
def category_products(id):
     p=Product.objects.filter(category=id)
     return len(p)




@register.filter(name='sub_category')
def sub_category(name):
     p=SubCategory.objects.all()
     return p


@register.filter(name='match_category')
def match_category(sub,cat):
     if sub.main_category.id==cat.id:
         return True
     else:
         return False
    



@register.filter(name='sub_category_orders')
def sub_category_orders(id):
     p=Orders.objects.filter(product__sub_category=id)
     return len(p)


@register.filter(name='discounted')
def discounted(id):
     p=Product.objects.filter(discount__gte=1)
     return len(p)


@register.filter(name='undiscounted')
def undiscounted(id):
     p=Product.objects.filter(discount=0)
     return len(p)

@register.filter(name='check_counter')
def check_counter(id):
     if id==20:
        return True
     return False

@register.filter(name='get_product_name')
def get_product_name(id):
    name=Product.objects.values_list("name",flat=True).get(id=id)
    return name

@register.filter(name='get_customer_email')
def get_customer_email(id):
    email=Customer.objects.values_list("email",flat=True).get(id=id)
    return email


@register.filter(name='city')
def city(id):
    cities=Orders.objects.values('city').order_by("city").annotate(dcount=Count('city'))
    
    result=[]
    if id=="city":
        for c in cities:
            result.append(c["city"])
        return result
    else:
        for c in cities:
            result.append(c["dcount"])
        return result


@register.filter(name='selling_products')
def selling_products(id):
    products=Orders.objects.values('product').order_by("product").annotate(dcount=Count('product'))
   
    products=products.order_by("-dcount")
    result=[]
    if id=="city":
        for c in products:
            result.append(c["product"])
        return result
    else:
        for c in products:
            result.append(c["dcount"])
        return result


@register.filter(name='least_selling_products')
def least_selling_products(id):
    products=Orders.objects.values('product').order_by("product").annotate(dcount=Count('product'))
    #print(products,type(products))
    products=products.order_by("dcount")
    result=[]
    if id=="city":
        for c in products:
            result.append(c["product"])
        return result
    else:
        for c in products:
            result.append(c["dcount"])
        return result


@register.filter(name='view_trends')
def view_trends(id):
    ord=Orders.objects.all().values()
    dataset = pd.DataFrame(ord)
    dataset['amount'] = dataset['quantity']*dataset['price']
    rfm_m = dataset.groupby('customer_id')['amount'].sum()
    rfm_m = rfm_m.reset_index()
    #dataset['order_date'] = pd.to_datetime(dataset['order_date'],format='%m/%d/%Y')
    #RFM implementation

    # Extracting amount by multiplying quantity and unit price and saving the data into amount variable.
    dataset["Amount"]  = dataset.quantity * dataset.price
    # Finding total amount spent per customer
    monetary = dataset.groupby("customer_id").Amount.sum()
    monetary = monetary.reset_index()
    monetary.head()
    #Frequency function

    # Getting the count of orders made by each customer based on customer ID.
    frequency = dataset.groupby("customer_id").order_id.count()
    frequency = frequency.reset_index()
    frequency.head()
    #creating master dataset
    master = monetary.merge(frequency, on = "customer_id", how = "inner")
    master.head()
    # Finding max data
    maximum = max(dataset.order_date)
    
    dataset['diff'] = maximum - dataset.order_date
    dataset.head()
    #Dataframe merging by recency
    recency = dataset.groupby('customer_id')['diff'].min()
    recency = recency.reset_index()
    recency.head()
    #Combining all recency, frequency and monetary parameters
    RFM = master.merge(recency, on = "customer_id")
    RFM.columns = ['customer_id','Amount','Frequency','Recency']
    RFM.head()
    # outlier treatment for Amount
    Q1 = RFM.Amount.quantile(0.25)
    Q3 = RFM.Amount.quantile(0.75)
    IQR = Q3 - Q1
    RFM = RFM[(RFM.Amount >= Q1 - 1.5*IQR) & (RFM.Amount <= Q3 + 1.5*IQR)]
    # outlier treatment for Frequency
    Q1 = RFM.Frequency.quantile(0.25)
    Q3 = RFM.Frequency.quantile(0.75)
    IQR = Q3 - Q1
    RFM = RFM[(RFM.Frequency >= Q1 - 1.5*IQR) & (RFM.Frequency <= Q3 + 1.5*IQR)]
    # outlier treatment for Recency
    Q1 = RFM.Recency.quantile(0.25)
    Q3 = RFM.Recency.quantile(0.75)
    IQR = Q3 - Q1
    RFM = RFM[(RFM.Recency >= Q1 - 1.5*IQR) & (RFM.Recency <= Q3 + 1.5*IQR)]
    # standardise all parameters
    RFM_norm1 = RFM.drop("customer_id", axis=1)
    RFM_norm1.Recency = RFM_norm1.Recency.dt.days
    standard_scaler = StandardScaler()
    RFM_norm1 = standard_scaler.fit_transform(RFM_norm1)
    RFM_norm1 = pd.DataFrame(RFM_norm1)
    RFM_norm1.columns = ['Frequency','Amount','Recency']
    RFM_norm1.head()

    
    def hopkins(X):
        d = X.shape[1]
        #d = len(vars) # columns
        n = len(X) # rows
        m = int(0.1 * n) 
        nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
    
        rand_X = sample(range(0, n, 1), m)
    
        ujd = []
        wjd = []
        for j in range(0, m):
            u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
            ujd.append(u_dist[0][1])
            w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
            wjd.append(w_dist[0][1])
    
        H = sum(ujd) / (sum(ujd) + sum(wjd))
        if isnan(H):
            #print(ujd, wjd)
            H = 0
    
        return H
    sse_ = []
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k).fit(RFM_norm1)
        sse_.append([k, silhouette_score(RFM_norm1, kmeans.labels_)])

    #plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])
    # sum of squared distances
    ssd = []
    for num_clusters in list(range(1,21)):
        model_clus = KMeans(n_clusters = num_clusters, max_iter=100)
        model_clus.fit(RFM_norm1)
        ssd.append(model_clus.inertia_)

    #plt.plot(ssd)
    # Kmeans with K=5
    model_clus5 = KMeans(n_clusters = 4, max_iter=50)
    model_clus5.fit(RFM_norm1)
    # analysis of clusters formed
    RFM.index = pd.RangeIndex(len(RFM.index))
    RFM_km = pd.concat([RFM, pd.Series(model_clus5.labels_)], axis=1)
    RFM_km.columns = ['customer_id', 'Amount', 'Frequency', 'Recency', 'ClusterID']
    if id=="data":
        return RFM_km.to_html()
    RFM_km.Recency = RFM_km.Recency.dt.days
    km_clusters_amount = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Amount.mean())
    km_clusters_frequency = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Frequency.mean())
    km_clusters_recency = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Recency.mean())
    df = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)
    df.columns = ["ClusterID", "Amount_mean", "Frequency_mean", "Recency_mean"]
    #df.head()
    
    #fig, axs = plt.subplots(1,3, figsize = (15,5))

    # sns.barplot(x=df.ClusterID, y=df.Amount_mean, ax = axs[0])
    # sns.barplot(x=df.ClusterID, y=df.Frequency_mean, ax = axs[1])
    # sns.barplot(x=df.ClusterID, y=df.Recency_mean, ax = axs[2])
    # plt.tight_layout()            
    #plt.show()
    
    if id=="amount":
        return df.Amount_mean
    if id=="frequency":
        return df.Frequency_mean
    if id=="recency":
        return df.Recency_mean
    return False