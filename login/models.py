from django.db import models
import datetime 
#from products.models import Orders


class Customer(models.Model):
        name=models.CharField(max_length=50)
        email=models.EmailField(max_length=254)
        GENDER_CHOICES = (
            ('M', 'Male'),
            ('F', 'Female'),
        )
        gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
        password=models.CharField(max_length=150)
        age=models.PositiveIntegerField()
        register_date=models.DateField(default=datetime.datetime.now().date())
        


        def email_exists(email):
            if Customer.objects.filter(email=email):
                return True

            return False

        @staticmethod
        def get_customer(email):
            try:
                return Customer.objects.get(email=email)
            except:
                return False

        def __str__(self):
            return self.email


class CustomerReport(Customer):
    class Meta:
        proxy=True
        verbose_name = 'Report'
        verbose_name_plural = 'Customer Reports'