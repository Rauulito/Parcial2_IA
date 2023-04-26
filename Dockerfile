FROM bitnami/pytorch:latest
ADD req.txt .
RUN pip install -r req.txt
WORKDIR /usr/src/predict
ADD predict.py .
ADD vehicles_model.pth .
CMD ["python", "predict.py"]
