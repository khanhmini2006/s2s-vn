package vn.s2s.robotclient.audio;

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.util.Log;

/**
 * Phát audio trả lời ra loa robot, lấy dữ liệu từ {@link PlaybackBuffer}.
 *
 * <p>16kHz mono PCM16 — bằng đúng định dạng server gửi về nên không phải chuyển đổi gì.
 *
 * <p>Luồng phát chạy liên tục kể cả lúc không có gì để phát: buffer cạn thì trả ra im
 * lặng ({@link PlaybackBuffer#read}), nên {@link AudioTrack} luôn có dữ liệu và không
 * bao giờ bị đói. Đây là lý do {@code talk.py:5-6} chọn cách này thay vì dừng/chạy lại
 * theo từng câu trả lời.
 */
public class SpeakerPlayer {

    private static final String TAG = "SpeakerPlayer";

    private static final int SAMPLE_RATE = MicRecorder.SAMPLE_RATE;

    /** 20ms mỗi lần ghi — đủ nhỏ để barge-in ngắt nhanh, đủ lớn để không tốn CPU. */
    private static final int GHI_BYTES = 640;

    private final PlaybackBuffer buffer;
    private volatile boolean dangChay;
    private Thread luongPhat;

    public SpeakerPlayer(PlaybackBuffer buffer) {
        this.buffer = buffer;
    }

    /** Mở loa và chạy vòng phát. */
    public void batDau() {
        if (dangChay) {
            return;
        }
        dangChay = true;
        luongPhat = new Thread(new Runnable() {
            @Override
            public void run() {
                vongPhat();
            }
        }, "loa-phat");
        luongPhat.start();
    }

    /** Đóng loa và xoá phần chưa phát. */
    public void dungLai() {
        dangChay = false;
        if (luongPhat != null) {
            try {
                luongPhat.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            luongPhat = null;
        }
        buffer.clear();
    }

    private void vongPhat() {
        int coToiThieu = AudioTrack.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (coToiThieu == AudioTrack.ERROR || coToiThieu == AudioTrack.ERROR_BAD_VALUE) {
            Log.e(TAG, "thiết bị không hỗ trợ phát 16kHz mono PCM16");
            dangChay = false;
            return;
        }

        int coBuffer = Math.max(coToiThieu, GHI_BYTES * 4);
        AudioTrack track = null;

        try {
            track = new AudioTrack(AudioManager.STREAM_MUSIC, SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    coBuffer, AudioTrack.MODE_STREAM);

            if (track.getState() != AudioTrack.STATE_INITIALIZED) {
                Log.e(TAG, "không khởi tạo được AudioTrack");
                return;
            }

            track.play();
            Log.i(TAG, "loa sẵn sàng: 16kHz mono, buffer " + coBuffer + " byte");

            byte[] khung = new byte[GHI_BYTES];
            while (dangChay) {
                // Cạn thì read() trả im lặng — write() luôn có dữ liệu, loa không đói.
                buffer.read(khung);
                int ghi = track.write(khung, 0, khung.length);
                if (ghi < 0) {
                    Log.e(TAG, "AudioTrack.write lỗi mã " + ghi);
                    break;
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "phát audio thất bại: " + e.getMessage());
        } finally {
            if (track != null) {
                try {
                    if (track.getPlayState() == AudioTrack.PLAYSTATE_PLAYING) {
                        track.stop();
                    }
                } catch (Exception ignored) {
                    // Thiết bị có thể đã ở trạng thái lỗi — vẫn phải release.
                }
                track.release();
            }
            dangChay = false;
            Log.i(TAG, "đã đóng loa");
        }
    }
}
