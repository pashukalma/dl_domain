apiVersion: v1
kind: Service
metadata:
  name: video-generation-service
  namespace: video-workflow
spec:
  selector:
    app: video-generation-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8001
  type: ClusterIP