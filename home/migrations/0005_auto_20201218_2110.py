# Generated by Django 3.1.1 on 2020-12-18 16:10

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0004_auto_20201213_0855'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='product',
            name='description',
        ),
        migrations.AlterField(
            model_name='carousel',
            name='published_date',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2020, 12, 18, 21, 10, 56, 154378)),
        ),
        migrations.AlterField(
            model_name='product',
            name='discount',
            field=models.PositiveIntegerField(default=0, verbose_name='Discount %'),
        ),
        migrations.AlterField(
            model_name='product',
            name='published_date',
            field=models.DateField(default=datetime.date(2020, 12, 18)),
        ),
    ]
