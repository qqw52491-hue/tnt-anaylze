#!/bin/bash
# Disposable Android process only. Never launches/stops the game or HUD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NDK="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/27.0.12077973}"
CC="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang"
ADB="${ADB:-/Applications/BlueStacks.app/Contents/MacOS/hd-adb}"
DEV="${ANDROID_SERIAL:-127.0.0.1:5555}"
OUT="$ROOT/target/tls-integration"
mkdir -p "$OUT"
bash "$ROOT/native/build-sniffer.sh"
"$CC" -std=gnu11 -Wall -Wextra -Werror -shared -fPIC "$ROOT/native/test_qt_resolver.c" \
    -Wl,-soname,libQt5Core_arm64-v8a.so -o "$OUT/libQt5Core_arm64-v8a.so" -ldl
"$CC" -std=gnu11 -Wall -Wextra -Werror -shared -fPIC "$ROOT/native/test_qt_local_caller.c" \
    -L"$OUT" -lQt5Core_arm64-v8a -o "$OUT/libqtlocalcaller.so"
"$CC" -std=gnu11 -Wall -Wextra -Werror "$ROOT/native/test_android_tls.c" \
    -ldl -o "$OUT/test-android-tls"
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$OUT/key.pem" -out "$OUT/cert.pem" \
    -days 1 -subj /CN=tnt-local-test >/dev/null 2>&1
cargo build --manifest-path "$ROOT/Cargo.toml" --bin net_live
python3 - "$ROOT" "$ADB" "$DEV" <<'PY'
import pathlib, socket, struct, subprocess, sys, threading, zlib, shlex, os
root = pathlib.Path(sys.argv[1]); adb = [sys.argv[2], '-s', sys.argv[3]]
out = root/'target/tls-integration'
remote = f'/data/local/tmp/tnt-tls-test-{os.getpid()}'
def run(*args, **kw): return subprocess.run([*adb, *args], check=True, **kw)
def shell(s, **kw): return run('shell', s, **kw)
def v(n):
    b=bytearray()
    while n > 127: b.append((n & 127)|128); n >>= 7
    return bytes(b)+bytes([n])
# Synthetic Action: the existing frozen test case decodes to world wind 7.1.
name=b'BattleNotify'
frame=b'HTTP/1.1 101 Switching Protocols\r\n\r\n'
for modern,raw in [(True,71),(False,449)]:
    wind=b'\x08\x02'+b''.join(b'\x10'+v(x) for x in [74108,0,73884])+b'\x18\x64\x20'+v(raw)
    body=b'\x08\x07\x10\x08\x42'+v(len(wind))+wind
    header=(struct.pack('>IH',2+len(name)+len(body),len(name)) if modern else
            struct.pack('<II',4+len(name)+len(body),len(name)))
    env=header+name+body
    payload=struct.pack('>I',len(env))+zlib.compress(env)
    assert len(payload)<126
    frame+=b'\x82'+bytes([len(payload)])+payload
(out/'frame.bin').write_bytes(frame)
base=shell('pm path com.grapefruit.game.tnt',capture_output=True,text=True).stdout.strip().splitlines()[0].removeprefix('package:')
libdir=str(pathlib.PurePosixPath(base).parent/'lib/arm64')
listener=socket.socket(); listener.bind(('127.0.0.1',19002)); listener.listen(); listener.settimeout(15)
chunks=[]; errors=[]
def receive():
    try:
        conn,_=listener.accept()
        with conn:
            conn.settimeout(10)
            while True:
                data=conn.recv(65536)
                if not data: break
                chunks.append(data)
    except Exception as e: errors.append(e)
worker=threading.Thread(target=receive,daemon=True)
reverse_created=False
directory_created=False
try:
    reverses=run('reverse','--list',capture_output=True,text=True).stdout
    assert not any('tcp:19002' in s.split() for s in reverses.splitlines()), 'test reverse port already in use'
    run('reverse','tcp:19002','tcp:19002')
    reverse_created=True
    shell('mkdir -p '+shlex.quote(remote))
    directory_created=True
    sniffer=pathlib.Path(os.environ.get('TNT_TEST_SNIFFER',str(root/'target/android/libtntsniff.so')))
    for src in [out/'libQt5Core_arm64-v8a.so',out/'libqtlocalcaller.so',out/'test-android-tls',out/'key.pem',out/'cert.pem',out/'frame.bin']:
        run('push',str(src),remote+'/'+src.name,capture_output=True)
    run('push',str(sniffer),remote+'/libtntsniff.so',capture_output=True)
    shell('chmod 700 '+shlex.quote(remote+'/test-android-tls'))
    worker.start()
    command=(f'LD_LIBRARY_PATH={shlex.quote(remote+":"+libdir)} LD_PRELOAD={shlex.quote(remote+"/libtntsniff.so")} '
             f'TNT_TUNNEL_PORT=19002 {shlex.quote(remote+"/test-android-tls")} '
             + ' '.join(shlex.quote(x) for x in [libdir+'/libssl_1_1.so',remote+'/cert.pem',remote+'/key.pem',remote+'/frame.bin',remote+'/libqtlocalcaller.so',remote]))
    try:
        result=shell(command,capture_output=True,text=True,timeout=20)
    except subprocess.CalledProcessError as e:
        print(e.stdout); print(e.stderr); raise
    print(result.stdout.strip()); print(result.stderr.strip())
    worker.join(12)
    assert not worker.is_alive() and not errors, errors
    raw=b''.join(chunks); (out/'raw.bin').write_bytes(raw)
    pos=0; kinds={}
    while pos+21<=len(raw):
        kind,pid,fd,ms,n=struct.unpack_from('<BIIQI',raw,pos)
        assert pos+21+n<=len(raw)
        kinds[kind]=kinds.get(kind,0)+1
        if kind in [11,12]: assert fd & 0x80000000
        pos+=21+n
    assert pos==len(raw) and kinds.get(11) and kinds.get(12), kinds
    replay=subprocess.run([str(root/'target/debug/net_live'),'replay',str(out/'raw.bin')],check=True,capture_output=True,text=True)
    assert 'WIND round=2' in replay.stdout and 'world=Some(7.1)' in replay.stdout, replay.stdout
    assert replay.stdout.count('WIND round=2')==4, 'both header formats must decode in both directions'
    assert replay.stdout.count('world=Some(7.1)')==4, 'updated direct/legacy XOR must both decode in both directions'
    print('Integration recording kinds:',kinds)
    print(replay.stdout.strip())
finally:
    listener.close()
    if reverse_created: run('reverse','--remove','tcp:19002',capture_output=True)
    # Only the unique directory created above, never a game/app directory.
    if directory_created: shell('rm -rf '+shlex.quote(remote))
PY
