from django.urls import path
from .views import Login, Signup
from . import views
from django.contrib.auth import views as auth_views
urlpatterns = [
    path('/', Login.as_view() ,name='login'),
    path('/signup/', Signup.as_view(), name='signup'),
    path('/logout/', views.logout),

    path("/password_reset/", auth_views.PasswordResetView.as_view(template_name='password_reset.html'),name="password_reset"),
    path("/password_reset/done/", auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name="password_reset_done"),
    path("/reset/<uidb64>/<token>/", auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html'), name="password_reset_confirm"),
    path("/password_reset_complete/", auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name="password_reset_complete"),
]
