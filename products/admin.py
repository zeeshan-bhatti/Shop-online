from django.contrib import admin
from .models import Category, Orders,Report,ViewTrends,Cluster
from django.contrib.admin.views.main import ChangeList
# Register your models here.
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'published_date')

class OrdersAdmin(admin.ModelAdmin):
    list_display = ('product', 'customer','city','quantity','price','order_date','payment_method','status')
    list_editable = ('status',)
    search_fields=('product__name','customer__email','city','status')
    list_filter=('order_date','status','product__category','product__sub_category','payment_method')
    actions=('deliver_selected_orders',)
    list_per_page = 50
    change_list_template='orders-graph.htm'
    date_hierarchy='order_date'
    readonly_fields=('product', 'customer','mobile','address','quantity','price','order_date','payment_method','order_time',
    'order_id','city')

   
    

    def has_add_permission(self, request, obj=None):
        return False
    def deliver_selected_orders(self,request,queryset):
        count=queryset.update(status=True)
        self.message_user(request,'{} orders are Deliverd Successfully!'.format(count))

class ReportAdmin(admin.ModelAdmin):
    change_list_template='reports.htm'
    

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


class ViewTrendsAdmin(admin.ModelAdmin):
    change_list_template='reports.htm'
    def get_changelist(self, request, **kwargs):
        return CustomChangeList

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

class ViewTrendsAdmin(admin.ModelAdmin):
    change_list_template='trends-graphs.htm'
    def get_changelist(self, request, **kwargs):
        return CustomChangeList

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


class ClusterAdmin(admin.ModelAdmin):
    change_list_template='clusters.htm'
    def get_changelist(self, request, **kwargs):
        return ClusterChangeList

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


class CustomChangeList(ChangeList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = 'Latest Trends'


class ClusterChangeList(ChangeList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = 'Latest Clusters'

admin.site.register(Category,CategoryAdmin)
admin.site.register(Orders,OrdersAdmin)
admin.site.register(Report,ReportAdmin)
admin.site.register(ViewTrends,ViewTrendsAdmin)
admin.site.register(Cluster,ClusterAdmin)