# M13 blocked disposition

## Exact blocker

M13 cannot execute its two required gates with the available authorities. The Golden 1K contract requires corpus `sha256:518a5213981de49f52fd3e18a880d73aee31eec86eb6dedaf8b39a3c6f7ab878` and manifest `sha256:7357fae12008fc3951f2ebfd819410c5c8d4612f12137d349031cd73a3186494`. The only available current real-teacher 1K volume has corpus `sha256:7719ed62c5bb8feedd7f7e955e52d0b373d8b09e3ec9f6b8256f99a8b5a7e9d1` and manifest `sha256:eea502c8e6adb33290342b3b39aba71a8ad19fc0f401182de61127d276b6b9be`. Running it against the frozen Golden would be an authority mismatch, not a parity test.

No genuine 10K corpus/config/provenance authority was found. The only 10K material is the explicitly derived synthetic M8 scaling fixture, which M13 forbids substituting.

## Work completed

- M12-integrated main authority and Contract pin verified.
- Golden fixture validation passed: 256 coordinates, semantic root `sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
- Real-teacher T4 smoke passed on the current 1K volume; no production parity or 10K run was claimed.
- All task-owned Modal apps stopped; no active tasks remain.

## Required external decision

Supply the frozen Golden source closure and a genuine 10K corpus/config/provenance closure, or explicitly revise M13 authority. This checkpoint does not invent either.
