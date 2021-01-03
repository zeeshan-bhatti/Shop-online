from django.contrib import admin
from .models import Customer,CustomerReport
# Register your models here.

class CustomerAdmin(admin.ModelAdmin):
    list_display=('name','email','gender')
    search_fields=('name','email')
    list_filter=('gender',)
    list_per_page = 50
    change_list_template = 'login-graphs.htm'

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False


class CustomerReportAdmin(admin.ModelAdmin):
    change_list_template='customer-reports.htm'
    

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

admin.site.register(Customer,CustomerAdmin)
admin.site.register(CustomerReport,CustomerReportAdmin)