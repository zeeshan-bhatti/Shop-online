from django.contrib import admin
from .models import Carousel, Product, Category, SubCategory


class CarouselAdmin(admin.ModelAdmin):
    list_display = ('slide_name', 'description')
    date_hierarchy='published_date'

class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'category','sub_category')
    search_fields=('name','category__name')
    list_filter=('published_date','category',)
    list_per_page = 50
    readonly_fields=("published_date",)
    change_list_template = 'change_list.htm'
    
    save_on_top = True
    

class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'published_date')
    search_fields=('name',)
    date_hierarchy='published_date'


class SubCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'published_date')
    search_fields=('name',)

    


admin.site.register(Carousel, CarouselAdmin),
admin.site.register(Product, ProductAdmin),
admin.site.register(SubCategory, SubCategoryAdmin),
