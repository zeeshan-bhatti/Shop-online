# Generated by Django 3.1.1 on 2020-12-18 16:10

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('products', '0004_auto_20201213_0855'),
    ]

    operations = [
        migrations.CreateModel(
            name='ViewTrends',
            fields=[
            ],
            options={
                'verbose_name': 'Trend',
                'verbose_name_plural': 'Trends',
                'proxy': True,
                'indexes': [],
                'constraints': [],
            },
            bases=('products.orders',),
        ),
        migrations.AlterField(
            model_name='orders',
            name='order_date',
            field=models.DateField(default=datetime.date(2020, 12, 18)),
        ),
        migrations.AlterField(
            model_name='orders',
            name='order_time',
            field=models.TimeField(default=datetime.time(21, 10, 56, 163811)),
        ),
    ]
