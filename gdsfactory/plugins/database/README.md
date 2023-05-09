# Database

Database is hosted on PlanetScale.
Blobs (gds files, yaml files) are hosted on S3.

## Environment variables

The following environment variables need to be set:

```
PS_DATABASE=gdslib
PS_HOST=xxx.xxxxxxx.xxxx.xxxxx
PS_USERNAME=
PS_PASSWORD=
PS_SSL_CERT=/etc/ssl/certs/ca-certificates.crt
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

## Tables

We currently have the following tables in our PlanetScale database:

- `GdsFile` (test table, not used in production)
- `Component`: Links a component through a hash to blobs (GDS, settings as yaml) in S3.
