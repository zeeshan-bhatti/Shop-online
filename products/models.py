from django.db import models
from home.models import Category,Product
from login.models import Customer

import datetime
# Create your models here.

class Orders(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    order_id= models.PositiveIntegerField()
    price = models.PositiveIntegerField()
    city = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    mobile=models.CharField(max_length=255)
    order_date = models.DateField(default=datetime.datetime.now().date())
    order_time = models.TimeField(default=datetime.datetime.now().time())
    status=models.BooleanField(default=False)
    PAYMENT_CHOICES = (
            ('cod', 'Cash on delivery'),
            ('online', 'Online'),
        )
    payment_method= models.CharField(max_length=10, choices=PAYMENT_CHOICES)
    

    class Meta:
        ordering = ('-order_date', )
        verbose_name = 'order'
        verbose_name_plural = 'Orders'
        

    

    @staticmethod
    def get_customer_orders(customer_id):
        return Orders.objects.filter(customer=customer_id).order_by('-order_date')


class Report(Orders):
    class Meta:
        proxy=True
        verbose_name = 'Report'
        verbose_name_plural = 'Reports'


class ViewTrends(Orders):
    class Meta:
        proxy=True
        verbose_name = 'Trend'
        verbose_name_plural = 'Trends'


class Cluster(Orders):
    class Meta:
        proxy=True
        verbose_name = 'Cluster'
        verbose_name_plural = 'Clusters'