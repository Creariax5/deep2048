from django.db import models

class Score(models.Model):
    name = models.CharField(max_length=200)
    score = models.IntegerField()
    size = models.IntegerField(default=4)
    date_publication = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name