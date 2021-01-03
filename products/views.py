from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.views import View
from home.models import Product, Category,SubCategory
from django.core.paginator import Paginator


class Products(View):
    def get(self,request):
        cart=request.session.get('cart')
        if not cart:
            request.session.cart={}

        products = Product.objects.all().order_by('-id')
        category = Category.objects.all
        sub_category=SubCategory.objects.all
        get_category=request.GET.get('category')
        if get_category:
            if get_category[0]=='0':
                products=Product.objects.filter(sub_category=get_category)
            else:
                products=Product.objects.filter(category=get_category)
        else:
            get_category=''
       
        # paginator=Paginator(products,per_page=6)
        # page_number=page_number = request.GET.get('page', 1)
        # page_obj = paginator.get_page(page_number)
        # if len(paginator.page_range)==1:
        #     flag=False
        # else:
        #     flag=True
        # products': page_obj.object_list,
        #                                         'page_number': int(page_number),
        # 'paginator':paginator,
                                                # 'flag':flag,
        return render(request, 'products.htm', {'products': products,
                                                  'category': category,
                                                  'cat':get_category,
                                                  'sub_category':sub_category,
                                                   })
    
    def post(self , request):
        if request.GET.get('category'):
            cat=request.GET.get('category')
        else:
            cat=''
        product = request.POST.get('product')
        remove=request.POST.get('remove')
        cart = request.session.get('cart')
        if cart:
            quantity = cart.get(product)
            if quantity:
                if remove:
                    if quantity<=1:
                        cart.pop(product)
                    else:
                        cart[product]  = quantity-1
                else:
                    cart[product]  = quantity+1

            else:
                cart[product] = 1
        else:
            cart = {}
            cart[product] = 1

        request.session['cart'] = cart
        return redirect(request.path+'?category='+cat)




       
