from celery import Celery

celery = Celery('tasks', broker='redis://localhost', backend='redis://localhost')
