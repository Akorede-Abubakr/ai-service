"""Render speech service entry point; use Render's injected PORT."""
import os
import pwd

import uvicorn


if __name__ == "__main__":
    if os.geteuid() == 0:
        account = pwd.getpwnam("asr")
        os.chown("/data", account.pw_uid, account.pw_gid)
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
    port = int(os.environ.get("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    uvicorn.run("app.main:create_app", factory=True, host="0.0.0.0", port=port,
                workers=1, access_log=False)
