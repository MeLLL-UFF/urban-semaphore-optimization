
import requests



url = 'https://atlas.microsoft.com/traffic/flow/segment/json'

params = {'api-version':'1.0',
          'style':'absolute',
          'zoom':'22',
          'query':'-22.907650,-43.111966',
          'subscription-key':'iis1njWZFFEhCifbmv17UPl2vD9OPkAeno_PF8bCRyo',
          'unit':'KMPH'
          }

headers = {'Authorization':'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Ik4tbEMwbi05REFMcXdodUhZbkhRNjNHZUNYYy'
                           'IsImtpZCI6Ik4tbEMwbi05REFMcXdodUhZbkhRNjNHZUNYYyJ9.eyJhdWQiOiJodHRwczovL2F0bGFzLm1pY3Jvc2'
                           '9mdC5jb20vIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvOTYwZWVlNjYtZjJjOC00YTIxLTkxN2MtMzN'
                           'hZTI4NzZhZTBjLyIsImlhdCI6MTU1NDA3MTUzNCwibmJmIjoxNTU0MDcxNTM0LCJleHAiOjE1NTQwNzU0MzQsImFj'
                           'ciI6IjEiLCJhaW8iOiJBVVFBdS84S0FBQUE1YlcxRGhneUpxekNhZmk5dUY2c2laZ3hvdG52NU1NNWpZUWFuZm9NN'
                           'GprKy9tQmpkMTBKM2xDc0xPcnRqK1BUVG90eW5Nc0x2QUNCM1NDS3FSb3dSQT09IiwiYWx0c2VjaWQiOiIxOmxpdm'
                           'UuY29tOjAwMDMwMDAwMUQxNUJEMjIiLCJhbXIiOlsicHdkIl0sImFwcGlkIjoiODQyYmM0NmYtMTVkNS00MWQyLWE'
                           '0YTctOTkwYjAyZDMyYjYxIiwiYXBwaWRhY3IiOiIxIiwiZW1haWwiOiJtYXJjZWxvLmRhbG1laWRhQG91dGxvb2su'
                           'Y29tIiwiZmFtaWx5X25hbWUiOiJkXHUwMDI3QWxtZWlkYSIsImdpdmVuX25hbWUiOiJNYXJjZWxvIiwiZ3JvdXBzI'
                           'jpbImMzZjk3YzkyLTcyMTAtNDQ1My04NWExLTdiMzFhNDM3MzU0MCJdLCJpZHAiOiJsaXZlLmNvbSIsImlwYWRkci'
                           'I6IjE3Ny40MS4zOS43MCIsIm5hbWUiOiJNYXJjZWxvIGRcdTAwMjdBbG1laWRhIiwib2lkIjoiOTZlYzUzNjgtMDR'
                           'jMC00YjY1LWE2NzgtNWIyNjA4YzgzMmE0IiwicHVpZCI6IjEwMDMyMDAwNDI4MzUyRjQiLCJzY3AiOiJ1c2VyX2lt'
                           'cGVyc29uYXRpb24iLCJzdWIiOiJBaFU3TTE2czI4aWluWmZGcTlVOTJTaFluY2c4N0JrNFloU3d5WlVMdlM4Iiwid'
                           'GlkIjoiOTYwZWVlNjYtZjJjOC00YTIxLTkxN2MtMzNhZTI4NzZhZTBjIiwidW5pcXVlX25hbWUiOiJsaXZlLmNvbS'
                           'NtYXJjZWxvLmRhbG1laWRhQG91dGxvb2suY29tIiwidXRpIjoiUnV0cU5LSXJpVXlZYmI2NUNhQkNBQSIsInZlciI'
                           '6IjEuMCJ9.gsW7O9OXrWTZ-OnGZw6sXkYH0w75lCTVjBF8jh7ZAzmCJHv0yyxcvqVMqqVlO8IK2rAlnuxT2owTM3m'
                           'PdO-C6PDU5c8umaEvCAT2H9KMEgbephXt2DevT9uHg1uDtPMECF-0OfxRxkACRC0kCKDKIFPLTveSrcFMrrFfmCgl'
                           'uPhXKdckOD6hb6tSgzXXx2QmZAx1zze94C_1aj7Du7ofsXK0vr1EfjlnyDQQKwnQeU3yFwZqagkpoXR7IAz0Wt2oc'
                           'w7bXUMb6ZcsxvwKDqe199ss9oTk4Ij3Szlu4T5YlDpXtIs1fk-ck6DfqSyBXJ777uDCEj6d8RrvYQNAcDEbjg',
           'x-ms-client-id':'iis1njWZFFEhCifbmv17UPl2vD9OPkAeno_PF8bCRyo'
           }

response = requests.get(url, params=params, headers=headers)

data = response.json()

flow_segment_data = data['flowSegmentData']

current_speed = flow_segment_data['currentSpeed']
free_flow_speed = flow_segment_data['freeFlowSpeed']
confidence = flow_segment_data['confidence']

print(current_speed)
print(free_flow_speed)
print(confidence)

