package vn.s2s.robotclient.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

/**
 * Dải cột hiển thị biên độ mic theo thời gian, cuộn từ phải sang trái.
 *
 * <p>Đây là thứ duy nhất trên màn hình trả lời được câu "mic có nghe thấy gì không".
 * Khi thiếu quyền {@code RECORD_AUDIO} hoặc mic bị tiến trình khác chiếm,
 * {@code AudioRecord} vẫn chạy và không báo lỗi — chỉ trả toàn số 0. Nhìn dải cột
 * phẳng lì là biết ngay, không cần mở logcat.
 *
 * <p>Cột dùng thang căn bậc hai chứ không tuyến tính: tai người nghe theo lôgarit, để
 * tuyến tính thì giọng nói bình thường chỉ nhúc nhích ở đáy.
 */
public class ThanhBienDo extends View {

    /** Số cột hiển thị; ~80ms mỗi cột nên 64 cột ≈ 5 giây gần nhất. */
    private static final int SO_COT = 64;

    private static final int BIEN_DO_TOI_DA = 32767;

    private final float[] mucCot = new float[SO_COT];
    private final Paint butCot = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint butDayNen = new Paint(Paint.ANTI_ALIAS_FLAG);

    private int viTriGhi;
    private boolean dangThu;

    public ThanhBienDo(Context context, AttributeSet attrs) {
        super(context, attrs);
        butCot.setStyle(Paint.Style.FILL);
        butDayNen.setStyle(Paint.Style.FILL);
        butDayNen.setColor(0xFF243040);
    }

    /** Bật/tắt trạng thái thu — đổi màu cột giữa hổ phách (đang nghe) và xám (nghỉ). */
    public void datDangThu(boolean dangThu) {
        this.dangThu = dangThu;
        postInvalidate();
    }

    /** Xoá sạch dải cột. */
    public void xoa() {
        java.util.Arrays.fill(mucCot, 0f);
        viTriGhi = 0;
        postInvalidate();
    }

    /**
     * Đẩy một giá trị biên độ đỉnh (0..32767) vào cột mới nhất.
     *
     * <p>Gọi được từ luồng thu, không cần chuyển về luồng UI.
     */
    public void themBienDo(int bienDoDinh) {
        float tyLe = Math.min(1f, Math.max(0f, bienDoDinh / (float) BIEN_DO_TOI_DA));
        mucCot[viTriGhi] = (float) Math.sqrt(tyLe); // thang căn bậc hai, hợp tai người
        viTriGhi = (viTriGhi + 1) % SO_COT;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int rong = getWidth();
        int cao = getHeight();
        if (rong <= 0 || cao <= 0) {
            return;
        }

        float rongCot = rong / (float) SO_COT;
        float dayCot = Math.max(1.5f, rongCot * 0.55f);
        float giua = cao / 2f;
        float caoToiDa = cao * 0.46f;

        butCot.setColor(dangThu ? 0xFFFFB020 : 0xFF3A4757);

        for (int i = 0; i < SO_COT; i++) {
            // Cột cũ nhất nằm bên trái, mới nhất bên phải.
            int chiSo = (viTriGhi + i) % SO_COT;
            float x = i * rongCot + rongCot / 2f;
            float nua = Math.max(dayCot / 2f, mucCot[chiSo] * caoToiDa);

            if (mucCot[chiSo] < 0.004f) {
                // Im lặng: chấm mảnh trên đường tâm, để thấy rõ dải vẫn đang chạy.
                canvas.drawRoundRect(x - dayCot / 2f, giua - dayCot / 2f,
                        x + dayCot / 2f, giua + dayCot / 2f, dayCot, dayCot, butDayNen);
            } else {
                canvas.drawRoundRect(x - dayCot / 2f, giua - nua,
                        x + dayCot / 2f, giua + nua, dayCot, dayCot, butCot);
            }
        }
    }
}
