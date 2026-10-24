"""One asyncio loop owns network/cache tasks outside the GL presentation thread."""
import asyncio
import threading


class IOLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name='stream-network', daemon=True)
        self.started = threading.Event()
        self.thread.start()
        self.started.wait()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.started.set()
        self.loop.run_forever()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.close()

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def close(self):
        if self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=10)
