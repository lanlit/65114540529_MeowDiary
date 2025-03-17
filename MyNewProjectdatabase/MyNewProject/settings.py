"""
Django settings for MyNewProject project.

Generated by 'django-admin startproject' using Django 5.0.6.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.0/ref/settings/
"""

import os
# from django.conf.urls.static import static
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-g*1d6o%ur4h1-8&k%(04z&%#@quj_&_nkmubabb_8#n0*7o!n2"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

#AUTH_USER_MODEL = 'MyApp.User'
AUTH_USER_MODEL = 'MyApp.CustomUser'


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "MyApp",
    "tailwind",
    "theme",
    # "django_extensions",
    
]

TAILWIND_APP_NAME = 'theme'

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    
]

ROOT_URLCONF = "MyNewProject.urls"

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

STATIC_ROOT = os.path.join(BASE_DIR, 'static')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# settings.py
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'MyApp/templates'],  # ระบุโฟลเดอร์เทมเพลต
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "MyApp/static"),  # อ้างอิงไปที่ static ของ MyApp
]


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # คำนวณ BASE_DIR ใหม่ให้เป็น string
# STATICFILES_DIRS = [
#     os.path.join(BASE_DIR, "MyApp", "static"),  # ใช้ os.path.join() แทน /
# ]

WSGI_APPLICATION = "MyNewProject.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'meowdiary_db',  # ชื่อฐานข้อมูล
        'USER': 'root',  # ชื่อผู้ใช้งาน MySQL ของคุณ
        'PASSWORD': '041078',  # รหัสผ่านของผู้ใช้งาน MySQL ของคุณ
        'HOST': 'localhost',  # หากคุณใช้ MySQL บนเครื่องของคุณเองให้ใช้ 'localhost'
        'PORT': '3306',  # พอร์ตเริ่มต้นของ MySQL คือ 3306
    }
}



# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = "th"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/


# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

INTERNAL_IPS = [
    "127.0.0.1",
]

NPM_BIN_PATH = "C:/Program Files/nodejs/npm"  # หรือพาธไปยัง npm บนเครื่องของคุณ

NPM_BIN_PATH = "C:/Program Files/nodejs/npm.cmd"

STATIC_URL = '/static/'

# STATICFILES_DIRS = [
#     BASE_DIR / "static",
# ]

LOGIN_REDIRECT_URL = '/home/' 
LOGOUT_REDIRECT_URL = '/login/'

AUTHENTICATION_BACKENDS = ['django.contrib.auth.backends.ModelBackend']

# EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
# EMAIL_HOST = 'smtp.example.com'  # เปลี่ยนเป็นโฮสต์ SMTP ที่คุณใช้
# EMAIL_PORT = 587
# EMAIL_USE_TLS = True
# EMAIL_HOST_USER = 'lanlalitsuwannasri@gmail.com'
# EMAIL_HOST_PASSWORD = 'Fernnever0410'
# DEFAULT_FROM_EMAIL = 'lanlalitsuwannasri@gmail.com'

EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = "lanlalitsuwannasri@gmail.com"
EMAIL_HOST_PASSWORD = "bjen lnzi bfeo fdzj"

CSRF_COOKIE_SECURE = False

