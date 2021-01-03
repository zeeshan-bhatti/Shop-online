from django.shortcuts import render, redirect, HttpResponseRedirect
from django.http import HttpResponse
from  django.contrib.auth.hashers import make_password,check_password
from .models import Customer
from home.models import Category
from django.views import View

class Login(View):
    
    return_url=None
    def get(self, request):
        if request.session.get('customer'):
            print('zeeshan bhatti')
            return redirect ('homepage')
        Login.return_url=request.GET.get('return_url')
        category = Category.objects.all
        return render(request, 'login.htm',{'category': category})

    def post(self,request):
        
        category = Category.objects.all
        email=request.POST.get('email')
        password=request.POST.get('password')
        customer=Customer.get_customer(email)
        error=None
        if customer:
            flag=check_password(password,customer.password)
            
            if flag:
                request.session['customer']=customer.id
                print(Login.return_url)
                if Login.return_url:
                    return HttpResponseRedirect(Login.return_url)
                else:
                    Login.return_url=None
                    return redirect('homepage')
            else:
                error='Invalid email or password'
                
        else:
            error='Invalid email or password'
        
        return render(request,'login.htm',{'error':error,
                                        'category':category})

    

class Signup(View):
    def get(self, request):
        if request.session.get('customer'):
            print('zeeshan bhatti')
            return redirect ('homepage')
        category = Category.objects.all
        return render(request, 'signup.htm',{'category': category})

    def post(self,request):
        category = Category.objects.all
        request_data=request.POST
        name=request_data.get('name')
        email=request_data.get('email')
        gender=request_data.get('gender')
        password=request_data.get('password')
        confirm_password=request_data.get('confirm_password')
        age=request_data.get('age')

        values={
            'name':name,
            'email':email,
            'gender':gender,
            'age':age
        }

        customer=Customer(name=name,
            email=email,
            gender=gender,
            password=password,
            age=age)
    
        error=self.validate_customer(name,email,password,confirm_password)
        data={
                'value':values,
                'error':error,
                'category':category
                
            }    
        
        if not error:
            customer.password=make_password(password)
            customer.save()
            request.session['customer']=customer.id
            return redirect('homepage')
        else:
            return render(request,'signup.htm',data)

    def validate_customer(self,name,email,password,confirm_password):
        error=None
        
        
        if len(name)< 4:
            error="Name must be atleast 4 characters long"
        elif Customer.email_exists(email):
            error="Email already used"
        elif len(password)<8:
            error="Password must be 8 characters long"
        elif password !=confirm_password:
            error="Password mismatch"
            

        return error

def logout(request):
    request.session.clear()
    return redirect('login')
 


def forget_password(request):
    if request.session.get('customer'):
            print('zeeshan bhatti')
            return redirect ('homepage')
    category = Category.objects.all
    return render(request, 'forget.htm',{'category': category})
