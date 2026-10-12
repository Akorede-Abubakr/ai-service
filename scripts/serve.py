"""Single process: one model copy, Render's PORT, no reload/access logs."""
import os
import pwd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    if os.geteuid() == 0:
        # Render mounts persistent disks at runtime, replacing image-layer
        # ownership. Make the mount root writable, then drop root before serving.
        account = pwd.getpwnam("ai")
        os.chown("/data", account.pw_uid, account.pw_gid)
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    uvicorn.run("app.main:create_render_app", factory=True, host="0.0.0.0", port=port,
                workers=1, access_log=False)
